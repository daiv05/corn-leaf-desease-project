# Mobile Handoff Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the documentation and traceability gaps found in a cross-repo audit against `maize-doctor-app` and `maize-doctor-api` — clarify the int8 model I/O contract, make exported artifacts verifiable (checksums), resolve the `MAIN_MODELS`-vs-`train.py` default ambiguity, and add a one-command way to push a freshly exported model into the app repo instead of a manual, error-prone copy.

**Architecture:** Additive changes only to `src/export/common.py` (new fields on already-written JSON artifacts) and a new `scripts/pipeline/sync_mobile_model.py` CLI that reads those artifacts and copies the two files `maize-doctor-app` actually bundles (`model_int8.tflite` + `labels.json`) into that repo's `assets/model/`, verifying the copy by hash and leaving a `manifest.json` behind for provenance. No changes to model architecture, training, or the export conversion logic itself.

**Tech Stack:** Python 3.11+, pytest, existing `src/export/` package.

**Spec:** `docs/es/deployment/react-native.md` (the mobile handoff contract this plan hardens), `CLAUDE.md` (pipeline conventions). Companion plans: `maize-doctor-app/docs/superpowers/plans/2026-08-16-fix-sync-client-and-remote-auth.md`, `maize-doctor-api/docs/superpowers/plans/2026-08-16-taxonomy-validation-and-docs.md` — none of the three plans depend on another to be individually testable, but all three originate from the same audit.

## Global Constraints

- Never hand-transcribe the class list anywhere; `class_to_idx` from the run's `summary.json` remains the only source of truth (unchanged from the existing `write_labels_json` docstring).
- `write_export_summary()` already writes `export_summary.json` (FP32) vs `export_summary_<quantize>.json` (quantized) into two separate files — new fields must go into both, not just one.
- The int8 TFLite/ONNX export in this pipeline is **weight-only** (dynamic, per-channel PTQ via `_quantize_pt2e`), so both FP32 and int8 variants keep float32 input/output tensors — this is already established fact from the prior `2026-08-15-finish-fase-8a-mobile-export.md` plan's Global Constraints, just not yet spelled out in the consumer-facing doc.
- Existing tests in `tests/export/test_common.py` use a `tmp_path`-based style with no real model/checkpoint — follow that convention for new tests instead of exporting a real model.

---

### Task 1: Clarify int8 input/output dtype in the mobile handoff doc

**Files:**
- Modify: `docs/es/deployment/react-native.md:37-51`

**Interfaces:**
- None (documentation-only).

- [ ] **Step 1: Edit the doc**

In `docs/es/deployment/react-native.md`, immediately after line 45 (`- El nombre del tensor de entrada en ONNX es \`input\`; en TFLite se accede por índice.`), insert:

```markdown
- **Esto aplica igual a la variante `int8`.** La cuantización de este pipeline es solo de pesos (dynamic per-channel PTQ), no de activaciones: los tensores de entrada/salida del grafo siguen siendo `float32` en ambas variantes, nunca `uint8`/`int8`. Si al cargar `model_int8.tflite` el runtime reporta un tensor de entrada no-float32, es un bug de exportación, no el comportamiento esperado.
```

And immediately after line 49 (`- Tensor \`float32\` de forma \`[1, 9]\` con **logits**, no probabilidades. La app debe aplicar softmax si quiere mostrar confianza.`), insert:

```markdown
- La salida también es `float32` en la variante `int8` por el mismo motivo (cuantización solo de pesos).
```

- [ ] **Step 2: Verify the doc still builds**

Run: `npm run docs:build`
Expected: build succeeds (exit 0), no VitePress dead-link/parse errors. This repo's docs build fails on broken links/markdown, so a clean exit is the acceptance check for a docs-only change.

- [ ] **Step 3: Commit**

```bash
git add docs/es/deployment/react-native.md
git commit -m "docs(export): clarify int8 variant keeps float32 model I/O"
```

---

### Task 2: Resolve the `MAIN_MODELS`-vs-`train.py` default ambiguity in CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (the "Principal (`train.py`)" bullet, currently starting `- **Principal (\`train.py\`):**`)

**Interfaces:**
- None (documentation-only).

- [ ] **Step 1: Edit the doc**

In `CLAUDE.md`, in the `## Pipelines` section, locate the bullet starting `- **Principal (\`train.py\`):** comparte toda la infraestructura de datos/modelos con baselines. Entrena una arquitectura (default \`shufflenet_v2_x1_0\`)...`. Immediately after that sentence's closing parenthesis for the dataset image count (`...31 623 antes)`), insert this clarifying sentence before the rest of the bullet continues:

```markdown
 Ojo: ese default de `shufflenet_v2_x1_0` es el de `train.py --models` cuando se omite el flag; `make train`/`make train-main` siempre pasan `--models $(MAIN_MODELS)` explícitamente, y `MAIN_MODELS` por defecto son **tres** modelos (`efficientnet_b0 shufflenet_v2_x1_0 efficientnet_lite0`, ver `Makefile`). Para entrenar solo `shufflenet_v2_x1_0` vía `make`, pasar `MAIN_MODELS=shufflenet_v2_x1_0` explícitamente.
```

- [ ] **Step 2: Verify the sentence landed correctly**

Run: `grep -n "MAIN_MODELS por defecto son" CLAUDE.md`
Expected: one match, inside the "Principal (`train.py`)" bullet.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: clarify train.py single-model default vs MAIN_MODELS Makefile default"
```

---

### Task 3: Add `schema_version` to `labels.json` and `sha256` to `export_summary*.json`

**Files:**
- Modify: `src/export/common.py:278-310` (`write_labels_json`)
- Modify: `src/export/common.py:392-446` (`write_export_summary`)
- Test: `tests/export/test_common.py`

**Interfaces:**
- Produces: `write_labels_json(...)` payload gains a `"schema_version": 1` key (all other keys/behavior unchanged; still raises `ValueError` on a non-contiguous `class_to_idx`).
- Produces: `write_export_summary(...)` payload's `formats[i]` dict gains a `"sha256": str | None` key (SHA-256 hex digest of `output_path`'s file contents, or `None` when `output_path` is `None`/the format failed).
- Consumes (Task 4 of this plan reads both): the new `sha256` field in `export_summary[_<quantize>].json` and the presence of `labels.json` alongside it.

- [ ] **Step 1: Write the failing tests**

Add to `tests/export/test_common.py`, right after `test_write_labels_json_ordena_por_indice`:

```python
def test_write_labels_json_incluye_schema_version(tmp_path):
    class_to_idx = {"healthy": 0, "common_rust": 1}

    output_path = write_labels_json(tmp_path, class_to_idx, "shufflenet_v2_x1_0", (224, 224))

    payload = json.loads(output_path.read_text())
    assert payload["schema_version"] == 1
```

And right after `test_write_export_summary_crea_export_dir`:

```python
def test_write_export_summary_incluye_sha256_del_artefacto(tmp_path):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    model_path = export_dir / "model.onnx"
    model_path.write_bytes(b"contenido de prueba")

    parity = ParityResult(
        format="onnx",
        n_samples=30,
        torch_top1_accuracy=0.9,
        exported_top1_accuracy=0.9,
        agreement_rate=1.0,
        max_abs_prob_diff=0.0001,
        mean_abs_prob_diff=0.00001,
        tolerance=1e-3,
        passed=True,
    )
    report = ExportReport(
        run_dir=tmp_path,
        model_name="shufflenet_v2_x1_0",
        formats=[
            ExportFormatResult(format="onnx", output_path=model_path, succeeded=True, parity=parity)
        ],
        library_versions={"torch": "2.12.1"},
    )

    write_export_summary(tmp_path, report)

    payload = json.loads((tmp_path / "export" / "export_summary.json").read_text())
    expected_sha256 = hashlib.sha256(b"contenido de prueba").hexdigest()
    assert payload["formats"][0]["sha256"] == expected_sha256


def test_write_export_summary_sha256_none_si_no_hay_output_path(tmp_path):
    report = ExportReport(
        run_dir=tmp_path,
        model_name="shufflenet_v2_x1_0",
        formats=[
            ExportFormatResult(format="tflite", output_path=None, succeeded=False, error="fallo")
        ],
        library_versions={},
    )

    write_export_summary(tmp_path, report)

    payload = json.loads((tmp_path / "export" / "export_summary.json").read_text())
    assert payload["formats"][0]["sha256"] is None
```

Add `import hashlib` to the top of `tests/export/test_common.py` alongside the existing `import json`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv\Scripts\pytest tests/export/test_common.py -v -k "schema_version or sha256"`
Expected: 3 failures — `KeyError: 'schema_version'` / `KeyError: 'sha256'`.

- [ ] **Step 3: Implement `schema_version` in `write_labels_json`**

In `src/export/common.py`, inside `write_labels_json` (around line 307), change:

```python
    payload = {"model": model_name, "image_size": list(image_size), "labels": labels}
```

to:

```python
    payload = {
        "schema_version": 1,
        "model": model_name,
        "image_size": list(image_size),
        "labels": labels,
    }
```

- [ ] **Step 4: Implement `sha256` in `write_export_summary`**

In `src/export/common.py`, add `import hashlib` to the top-level imports (alongside `import json`). Then add this helper just above `write_export_summary` (after the closing of `_export_single_format`, before line 392):

```python
def _sha256_file(path: Path) -> str:
    """
    Calcula el digest SHA-256 de un archivo, leyendo en bloques para no cargarlo entero en memoria.

    @param {Path} path Archivo a hashear.
    @returns {str} Digest hexadecimal.
    """
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

Then, inside `write_export_summary`'s `payload["formats"]` list comprehension, add the `sha256` key alongside the existing `parity` key:

```python
        "formats": [
            {
                "format": f.format,
                "output_path": (
                    str(f.output_path.relative_to(run_dir)) if f.output_path else None
                ),
                "succeeded": f.succeeded,
                "error": f.error,
                "sha256": _sha256_file(f.output_path) if f.output_path else None,
                "parity": (
```
(leave the rest of the `parity` block exactly as-is)

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv\Scripts\pytest tests/export/test_common.py -v`
Expected: all tests in the file PASS, including the 3 new ones and the pre-existing ones (no regressions).

- [ ] **Step 6: Update the mobile handoff doc's artifact table**

In `docs/es/deployment/react-native.md`, update line 19 (the `labels.json` row) to:

```markdown
| `labels.json` | Orden de clases del modelo: `{"schema_version": int, "model": str, "image_size": [h, w], "labels": list[str]}`, donde `labels[i]` es el nombre de clase del índice de salida `i`. Se escribe una sola vez por run, no por formato ni por variante de cuantización — el orden de clases no cambia entre ellos |
```

And update line 17 (the `export_summary.json` row) to:

```markdown
| `export_summary.json` | Resultado de la conversión y la paridad numérica. Cada entrada de `formats[]` incluye `sha256` del archivo exportado — usarlo para verificar que una copia (por ejemplo, la que se empaqueta en la app) no se corrompió ni quedó desactualizada |
```

- [ ] **Step 7: Verify the doc still builds**

Run: `npm run docs:build`
Expected: build succeeds (exit 0).

- [ ] **Step 8: Commit**

```bash
git add src/export/common.py tests/export/test_common.py docs/es/deployment/react-native.md
git commit -m "feat(export): add schema_version to labels.json and sha256 to export_summary"
```

---

### Task 4: `scripts/pipeline/sync_mobile_model.py` — verified copy into `maize-doctor-app`

**Files:**
- Create: `scripts/pipeline/sync_mobile_model.py`
- Test: `tests/pipeline/test_sync_mobile_model.py`
- Modify: `Makefile` (new `sync-mobile-model` target, near the `export-main`/`eval-export-main` section)

**Interfaces:**
- Consumes: `export_summary.json`/`export_summary_<quantize>.json`'s `formats[i].sha256` field and `output_path`, and the `_sha256_file(path: Path) -> str` helper (both from Task 3 of this plan — imported from `src.export.common`, not redefined).
- Produces: `sync_mobile_model(run_dir: Path, dest_dir: Path, *, fmt: str = "tflite", quantize: str | None = "int8") -> Path` — copies `run_dir/export/model[_<quantize>].<fmt>` and `run_dir/export/labels.json` into `dest_dir`, verifies the copied model file's SHA-256 against the value recorded in the matching `export_summary*.json`, writes `dest_dir/manifest.json`, and returns its path. Raises `ValueError` if the recorded and copied hashes don't match (corrupt copy) or if the expected format entry is missing from the summary.

- [ ] **Step 1: Write the failing test**

Create `tests/pipeline/test_sync_mobile_model.py`:

```python
import hashlib
import json
from pathlib import Path

import pytest

from scripts.pipeline.sync_mobile_model import sync_mobile_model


def _make_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "outputs" / "main" / "shufflenet_v2_x1_0" / "20260816_120000"
    export_dir = run_dir / "export"
    export_dir.mkdir(parents=True)

    model_bytes = b"modelo tflite de prueba"
    (export_dir / "model_int8.tflite").write_bytes(model_bytes)
    labels_payload = {
        "schema_version": 1,
        "model": "shufflenet_v2_x1_0",
        "image_size": [224, 224],
        "labels": ["common_rust", "healthy"],
    }
    (export_dir / "labels.json").write_text(json.dumps(labels_payload))

    summary_payload = {
        "run_id": run_dir.name,
        "model": "shufflenet_v2_x1_0",
        "exported_at": "2026-08-16T12:00:00",
        "quantize": "int8",
        "library_versions": {},
        "formats": [
            {
                "format": "tflite",
                "output_path": "export/model_int8.tflite",
                "succeeded": True,
                "error": None,
                "sha256": hashlib.sha256(model_bytes).hexdigest(),
                "parity": None,
            }
        ],
    }
    (export_dir / "export_summary_int8.json").write_text(json.dumps(summary_payload))
    return run_dir


def test_sync_mobile_model_copia_y_escribe_manifest(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    dest_dir = tmp_path / "app_assets"

    manifest_path = sync_mobile_model(run_dir, dest_dir, fmt="tflite", quantize="int8")

    assert (dest_dir / "model_int8.tflite").read_bytes() == b"modelo tflite de prueba"
    assert (dest_dir / "labels.json").exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["run_id"] == "20260816_120000"
    assert manifest["model"] == "shufflenet_v2_x1_0"
    assert manifest["format"] == "tflite"
    assert manifest["quantize"] == "int8"
    assert manifest["sha256"] == hashlib.sha256(b"modelo tflite de prueba").hexdigest()


def test_sync_mobile_model_detecta_hash_incorrecto(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    summary_path = run_dir / "export" / "export_summary_int8.json"
    summary = json.loads(summary_path.read_text())
    summary["formats"][0]["sha256"] = "0" * 64
    summary_path.write_text(json.dumps(summary))
    dest_dir = tmp_path / "app_assets"

    with pytest.raises(ValueError, match="no coincide"):
        sync_mobile_model(run_dir, dest_dir, fmt="tflite", quantize="int8")


def test_sync_mobile_model_formato_ausente_en_summary(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    dest_dir = tmp_path / "app_assets"

    with pytest.raises(ValueError, match="No se encontro"):
        sync_mobile_model(run_dir, dest_dir, fmt="onnx", quantize="int8")
```

Create `tests/pipeline/__init__.py` (empty file) if the directory doesn't already have one.

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv\Scripts\pytest tests/pipeline/test_sync_mobile_model.py -v`
Expected: `ModuleNotFoundError: No module named 'scripts.pipeline.sync_mobile_model'`.

- [ ] **Step 3: Implement `sync_mobile_model`**

Create `scripts/pipeline/sync_mobile_model.py`:

```python
"""Copia el modelo exportado y labels.json al repo de la app, con verificacion de hash.

Lee `<run_dir>/export/export_summary[_<quantize>].json` para obtener el sha256
registrado del artefacto (ver `src/export/common.py::write_export_summary`), copia
`model[_<quantize>].<fmt>` y `labels.json` a `dest_dir`, re-calcula el hash de la copia
y aborta si no coincide. Deja `dest_dir/manifest.json` como registro de procedencia,
para que quede trazable de que run/hash viene el modelo que trae la app empaquetado.
"""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.export.common import _sha256_file


def _summary_path(run_dir: Path, quantize: str | None) -> Path:
    export_dir = run_dir / "export"
    name = "export_summary.json" if quantize is None else f"export_summary_{quantize}.json"
    return export_dir / name


def sync_mobile_model(
    run_dir: Path, dest_dir: Path, *, fmt: str = "tflite", quantize: str | None = "int8"
) -> Path:
    """
    Copia el modelo exportado y labels.json a dest_dir, verificando el hash del artefacto.

    @param {Path} run_dir Directorio del run (contiene `export/`).
    @param {Path} dest_dir Directorio destino (ej: `<maize-doctor-app>/assets/model`).
    @param {str} fmt Formato a copiar ("onnx" o "tflite").
    @param {str|None} quantize Variante ("int8" o None para FP32).
    @returns {Path} Ruta del `manifest.json` escrito en `dest_dir`.
    @throws {ValueError} Si el formato no aparece en el summary, o si el hash de la
        copia no coincide con el registrado (copia corrupta o summary desactualizado).
    """
    summary_path = _summary_path(run_dir, quantize)
    summary = json.loads(summary_path.read_text())

    match = next((f for f in summary["formats"] if f["format"] == fmt), None)
    if match is None or not match["succeeded"]:
        raise ValueError(
            f"No se encontro un export exitoso de formato '{fmt}' en {summary_path}"
        )

    export_dir = run_dir / "export"
    model_filename = "model.tflite" if quantize is None and fmt == "tflite" else None
    model_filename = model_filename or Path(match["output_path"]).name
    source_model = export_dir / model_filename
    source_labels = export_dir / "labels.json"

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_model = dest_dir / model_filename
    dest_labels = dest_dir / "labels.json"
    shutil.copy2(source_model, dest_model)
    shutil.copy2(source_labels, dest_labels)

    copied_sha256 = _sha256_file(dest_model)
    if copied_sha256 != match["sha256"]:
        raise ValueError(
            f"El hash de la copia ({copied_sha256}) no coincide con el registrado en "
            f"{summary_path} ({match['sha256']})"
        )

    manifest = {
        "run_id": summary["run_id"],
        "model": summary["model"],
        "format": fmt,
        "quantize": quantize,
        "sha256": copied_sha256,
        "source_run_dir": str(run_dir),
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = dest_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copia el modelo exportado y labels.json al repo de la app movil."
    )
    parser.add_argument("--run-dir", required=True, dest="run_dir")
    parser.add_argument("--dest", required=True, help="Directorio destino (assets/model de la app).")
    parser.add_argument("--format", default="tflite", choices=["onnx", "tflite"])
    parser.add_argument(
        "--quantize", default="int8", help="'int8' o 'none' (default: int8)."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    quantize = None if args.quantize.lower() in {"none", ""} else args.quantize
    manifest_path = sync_mobile_model(
        Path(args.run_dir), Path(args.dest), fmt=args.format, quantize=quantize
    )
    print(f"Sincronizado. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv\Scripts\pytest tests/pipeline/test_sync_mobile_model.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Add a Makefile target**

In `Makefile`, in the export section (near the `export-main`/`eval-export-main` targets), add:

```makefile
.PHONY: sync-mobile-model
# Copia el modelo exportado + labels.json a maize-doctor-app/assets/model, con
# verificacion de hash. RUN_DIR y DEST son obligatorios.
sync-mobile-model:
	$(PYTHON) -m scripts.pipeline.sync_mobile_model \
		--run-dir $(RUN_DIR) \
		--dest $(DEST) \
		$(if $(FORMAT),--format $(FORMAT),) \
		$(if $(QUANTIZE),--quantize $(QUANTIZE),)
```

- [ ] **Step 6: Run the full export test suite to check for regressions**

Run: `venv\Scripts\pytest tests/export/ tests/pipeline/ -v`
Expected: all PASS, no regressions from Task 3's changes.

- [ ] **Step 7: Commit**

```bash
git add scripts/pipeline/sync_mobile_model.py tests/pipeline/test_sync_mobile_model.py Makefile
git commit -m "feat(export): add sync_mobile_model to verifiably copy exports into the app repo"
```
