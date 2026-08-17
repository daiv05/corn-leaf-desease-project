# Finish Fase 8a — Mobile Export Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce validated TFLite artifacts (FP32 + int8) and a machine-readable `labels.json` for the three already-trained `main` models, and stage a benchmarking candidate set for the `maize-doctor-app` repo's Fase 8b.

**Architecture:** Extend the existing, already-tested export pipeline (`src/export/`, `scripts/pipeline/export.py`) with a small `labels.json` writer sourced from each run's `summary.json` `class_to_idx` — the same source of truth `export.py` already reads, so the app never has to hand-transcribe class order again. Then run the existing `modal-export-main`/`modal-eval-export-main` Make targets (TFLite export requires Linux/`litert-torch`, unavailable on this Windows machine, so it must run on Modal) against the three runs that already exist and already passed ONNX parity, pull the artifacts locally, and stage all three int8 candidates for the app's on-device benchmark rather than guessing the final pick now.

**Tech Stack:** Python 3, PyTorch, `litert-torch` (TFLite export, Linux-only), Modal (GPU cloud), pytest, existing Makefile targets.

**Spec:** `docs/es/deployment/react-native.md` (model I/O contract), `CLAUDE.md` (pipeline conventions), and the maize-doctor-app repo's `implementation-plan.md` Fase 8a section (a corrected copy of its assumptions is captured in the Global Constraints below — the original doc predates this pipeline's actual export code and assumes a manual `ai-edge-torch` conversion + static calibration-based Int8 quantization that this project does not actually use).

## Global Constraints

- Never hardcode class order anywhere downstream — always derive it from `class_to_idx` in the run's `summary.json` (this is the exact bug already found live in the app; see the companion app-side plan).
- **Correction to `implementation-plan.md`'s assumptions:** this pipeline's Int8 TFLite export (`src/export/tflite_export.py`, `_quantize_pt2e`) is **dynamic, per-channel, weights-only** quantization — not static full-integer PTQ with a calibration set. The graph's external input/output tensors stay **float32** in both FP32 and Int8 exports (confirmed by `docs/es/deployment/react-native.md`'s contract section, which states the output is float32 logits unconditionally). Consequently there is no runtime scale/zero-point to read off the model for either format — the app-side plan must not build that machinery.
- TFLite export needs `litert-torch`, which only supports Linux → always use `make modal-export-main` / `make modal-eval-export-main`, never the local `make export-main` on this Windows checkout.
- Quantized-vs-FP32 parity tolerance is intentionally relaxed for Int8 (`tolerance=0.15`, `min_agreement_rate=0.95` vs `1e-3`/`1.0` for FP32) — do not treat an Int8 parity "pass" as equivalent evidence to an FP32 pass; that's what `eval-export-main`'s real test-set macro-F1 is for.
- Model size ≤ 20 MB and inference latency ≤ 300 ms on a Snapdragon 6xx-class device are the hard acceptance constraints (`model-ml.md`). Size is checkable here; latency is only checkable on-device in the app repo's Fase 8b — this plan does not claim a final model choice, only prepares candidates for that benchmark.
- All three candidate runs (`efficientnet_b0`, `efficientnet_lite0`, `shufflenet_v2_x1_0`) already exist under `outputs-remote/main/<model>/<run_id>/`, already have `best.pth` + `summary.json` + a **passing ONNX FP32 parity** (`agreement_rate: 1.0`), and already clear the Macro F1 ≥ 0.85 target (0.943 / 0.947 / 0.924 respectively, from each run's `test_grouped_metrics.json`). Do not retrain — this plan only finishes exporting/labeling/staging what's already trained.

---

### Task 1: `labels.json` export alongside the model artifacts

**Files:**
- Modify: `src/export/common.py` (add `write_labels_json`)
- Modify: `scripts/pipeline/export.py` (call it from `_export_one`)
- Modify: `src/export/__init__.py` (export the new symbol)
- Test: `tests/export/test_common.py`

**Interfaces:**
- Produces: `write_labels_json(run_dir: Path, class_to_idx: dict[str, int], model_name: str, image_size: tuple[int, int]) -> Path` — writes `<run_dir>/export/labels.json` with `{"model": str, "image_size": [h, w], "labels": list[str]}`, where `labels[i]` is the class name whose `class_to_idx` value is `i`. Later tasks (and the app repo) read this file directly instead of re-deriving class order.

- [x] **Step 1: Write the failing test**

Add to `tests/export/test_common.py` (alongside the existing `write_export_summary` tests, using the same `tmp_path`-based style already in that file):

```python
def test_write_labels_json_ordena_por_indice(tmp_path):
    from src.export.common import write_labels_json

    class_to_idx = {"healthy": 2, "common_rust": 0, "fall_armyworm": 1}

    output_path = write_labels_json(tmp_path, class_to_idx, "shufflenet_v2_x1_0", (224, 224))

    assert output_path == tmp_path / "export" / "labels.json"
    payload = json.loads(output_path.read_text())
    assert payload["model"] == "shufflenet_v2_x1_0"
    assert payload["image_size"] == [224, 224]
    assert payload["labels"] == ["common_rust", "fall_armyworm", "healthy"]
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/export/test_common.py::test_write_labels_json_ordena_por_indice -v`
Expected: FAIL with `ImportError: cannot import name 'write_labels_json'`.

- [x] **Step 3: Write minimal implementation**

In `src/export/common.py`, add after `write_export_summary` (reuses the same `export_dir` pattern already used there):

```python
def write_labels_json(
    run_dir: Path,
    class_to_idx: dict[str, int],
    model_name: str,
    image_size: tuple[int, int],
) -> Path:
    """
    Persiste <run_dir>/export/labels.json con el orden de clases del modelo.

    Existe para que ningun consumidor del modelo exportado (la app movil, un script de
    evaluacion externo) tenga que re-derivar o hardcodear el orden de clases: lo lee de
    aqui. Un desajuste de orden entre este archivo y el modelo real es indetectable en
    runtime (el modelo igual devuelve 9 logits validos), asi que la unica fuente de verdad
    aceptable es `class_to_idx` de `summary.json`, nunca una lista transcrita a mano.

    @param {Path} run_dir Directorio del run.
    @param {dict[str,int]} class_to_idx Mapeo clase->indice con el que se entreno/exporto.
    @param {str} model_name Nombre del modelo (metadata informativa).
    @param {tuple[int,int]} image_size Alto y ancho de entrada esperado (h, w).
    @returns {Path} Ruta del archivo escrito.
    """
    export_dir = run_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    labels = [idx_to_class[i] for i in range(len(idx_to_class))]
    payload = {"model": model_name, "image_size": list(image_size), "labels": labels}
    output_path = export_dir / "labels.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return output_path
```

In `src/export/__init__.py`, add `write_labels_json` to both the `from src.export.common import (...)` block and `__all__`.

In `scripts/pipeline/export.py`, import `write_labels_json` alongside the existing `src.export.common` imports, and call it in `_export_one` right after `class_to_idx, _, image_size = resolve_export_inputs(...)`:

```python
    class_to_idx, _, image_size = resolve_export_inputs(run_dir, model_name, config_path)
    write_labels_json(run_dir, class_to_idx, model_name, image_size)
    device = select_device()
```

- [x] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/export/test_common.py -v`
Expected: PASS, including the new test and all pre-existing tests in that file (no regressions).

- [x] **Step 5: Commit**

```bash
git add src/export/common.py src/export/__init__.py scripts/pipeline/export.py tests/export/test_common.py
git commit -m "feat(export): write labels.json with class order alongside exported models"
```

---

### Task 2: Export TFLite (FP32 + Int8) for all three trained models via Modal

**Files:** none (operational task — runs the pipeline built in Task 1 plus the pre-existing `modal-export-main` target against the Modal `corn-outputs` volume).

**Interfaces:**
- Consumes: `write_labels_json` from Task 1 (runs automatically as part of `export.py`).
- Produces: for each of `efficientnet_b0`, `efficientnet_lite0`, `shufflenet_v2_x1_0`, on the Modal volume under `outputs/main/<model>/<run_id>/export/`: `model.tflite`, `model_int8.tflite`, `labels.json`, `export_summary.json`, `export_summary_int8.json`.

- [x] **Step 1: Push the Task 1 code change and export FP32 TFLite for all three models**

Run: `make modal-export-main MAIN_MODELS="efficientnet_b0 efficientnet_lite0 shufflenet_v2_x1_0" EXPORT_FORMATS=tflite`

Expected: for each of the 3 models, stdout ends with `[OK] tflite -> export/model.tflite` and a parity line reading `paridad: paso (... tolerance=0.001000 ...)`. If `[FALLO]` appears for any model, stop and read the printed `error:` line before continuing — do not proceed to Int8 export for a model whose FP32 export failed.

- [x] **Step 2: Export Int8 TFLite for all three models**

Run: `make modal-export-main MAIN_MODELS="efficientnet_b0 efficientnet_lite0 shufflenet_v2_x1_0" EXPORT_FORMATS=tflite QUANTIZE=int8`

Expected: same as Step 1, but tolerance in the printed parity line reads `tolerance=0.150000` (the relaxed Int8 default from `_PARITY_DEFAULTS`) and writes to `export/model_int8.tflite` / `export/export_summary_int8.json`.

- [x] **Step 3: Pull the artifacts locally**

Run: `make modal-pull`

Expected: `outputs-remote/main/<model>/<run_id>/export/` now contains `model.tflite`, `model_int8.tflite`, `labels.json`, `export_summary.json`, `export_summary_int8.json` for all three models. Verify with:

```bash
for m in efficientnet_b0 efficientnet_lite0 shufflenet_v2_x1_0; do
  find "outputs-remote/main/$m" -path "*/export/*" \( -name "model.tflite" -o -name "model_int8.tflite" -o -name "labels.json" \)
done
```

Expected: 9 lines (3 files × 3 models), all present.

- [x] **Step 4: Sanity-check `labels.json` matches the documented contract**

Run: `cat outputs-remote/main/efficientnet_b0/*/export/labels.json`

Expected output (order must match exactly — this is the file the app repo's Fase 8b plan reads instead of hardcoding class order):

```json
{
  "model": "efficientnet_b0",
  "image_size": [224, 224],
  "labels": ["common_rust", "fall_armyworm", "gray_leaf_spot", "healthy", "lethal_necrosis", "nitrogen_deficiency", "northern_corn_leaf_blight", "phosphorus_deficiency", "potassium_deficiency"]
}
```

(No commit — this task only produces artifacts under the git-ignored `outputs`/`outputs-remote` trees.)

---

### Task 3: Evaluate exported Int8 artifacts on the full test set

**Files:** none (operational task).

**Interfaces:**
- Consumes: `model_int8.tflite` from Task 2.
- Produces: `outputs-remote/main/<model>/<run_id>/export/eval_tflite_int8.json` (+ per-image CSV) for each of the 3 models — real macro-F1 and per-class breakdown of the quantized artifact, as opposed to the ~30-sample numeric parity check from Task 2.

> **Estado (2026-08-17): COMPLETADA.** Evaluacion corrida en Modal sobre el test set completo (5 015 imagenes). `efficientnet_b0` supera el umbral de `--max-macro-f1-drop` (cae 0.0186 vs 0.01 permitido) y quedo registrado con warning; los otros dos pasan sin observaciones. Resultados en la tabla de la Task 4, Step 2.

- [x] **Step 1: Run the evaluation for all three Int8 TFLite artifacts**

Run: `make modal-eval-export-main MAIN_MODELS="efficientnet_b0 efficientnet_lite0 shufflenet_v2_x1_0" EXPORT_FORMATS=tflite QUANTIZE=int8`

Expected: completes without raising (default `--max-macro-f1-drop` is `0.01`; a model whose Int8 macro-F1 drops more than that from its own PyTorch baseline will make this command exit non-zero — if that happens, note which model failed and by how much, but do not attempt to fix quantization in this task, just record it for the model-selection comparison in Task 4).

- [x] **Step 2: Pull results and read the macro-F1 delta per model**

Run: `make modal-pull`, then for each model:

```bash
cat outputs-remote/main/efficientnet_b0/*/export/eval_tflite_int8.json
cat outputs-remote/main/efficientnet_lite0/*/export/eval_tflite_int8.json
cat outputs-remote/main/shufflenet_v2_x1_0/*/export/eval_tflite_int8.json
```

Expected: each JSON contains the exported artifact's real accuracy/macro-F1 over the full test split plus a delta against the PyTorch baseline. Record the three `macro_f1` values and their deltas — Task 4 uses these numbers directly.

(No commit — evaluation outputs live under the git-ignored `outputs`/`outputs-remote` trees.)

---

### Task 4: Stage the three Int8 candidates for the app repo's on-device benchmark

**Files:**
- Create (in the **`maize-doctor-app`** repo, not this one): `assets/model/candidates/efficientnet_b0/model_int8.tflite`, `assets/model/candidates/efficientnet_lite0/model_int8.tflite`, `assets/model/candidates/shufflenet_v2_x1_0/model_int8.tflite`, `assets/model/labels.json`

**Interfaces:**
- Consumes: `model_int8.tflite` (×3) and `labels.json` from Task 2/3 (`labels.json` is identical across all three models since they share `config/dataset.yaml` — copy any one of the three, they're byte-identical in content).
- Produces: the exact input the app repo's Fase 8b plan (`maize-doctor-app/docs/superpowers/plans/2026-08-15-fase-8b-tflite-integration.md`, Task 1) expects to find under `assets/model/`.

Why three candidates instead of picking one now: `model-ml.md`'s hard latency constraint (≤300 ms) can only be measured on the Snapdragon 6xx-class reference device, not from this desktop pipeline — Task 3's macro-F1 numbers are necessary but not sufficient to choose. Bundling all three behind a dev-only picker lets the app repo's benchmark task (see companion plan) measure real on-device latency/size before permanently committing to one, then delete the other two before release.

- [x] **Step 1: Copy the three Int8 artifacts and one `labels.json` into the app repo**

```bash
mkdir -p ../maize-doctor-app/assets/model/candidates/efficientnet_b0
mkdir -p ../maize-doctor-app/assets/model/candidates/efficientnet_lite0
mkdir -p ../maize-doctor-app/assets/model/candidates/shufflenet_v2_x1_0
cp outputs-remote/main/efficientnet_b0/*/export/model_int8.tflite ../maize-doctor-app/assets/model/candidates/efficientnet_b0/model_int8.tflite
cp outputs-remote/main/efficientnet_lite0/*/export/model_int8.tflite ../maize-doctor-app/assets/model/candidates/efficientnet_lite0/model_int8.tflite
cp outputs-remote/main/shufflenet_v2_x1_0/*/export/model_int8.tflite ../maize-doctor-app/assets/model/candidates/shufflenet_v2_x1_0/model_int8.tflite
cp outputs-remote/main/efficientnet_b0/*/export/labels.json ../maize-doctor-app/assets/model/labels.json
```

Expected: `ls -la ../maize-doctor-app/assets/model/candidates/*/model_int8.tflite` shows 3 files sized approximately 4.4 MB (`efficientnet_b0`), 3.6 MB (`efficientnet_lite0`), 1.4 MB (`shufflenet_v2_x1_0`) — matching the sizes documented in `docs/es/deployment/react-native.md`'s artifact table. All three are individually ≤ 20 MB.

- [x] **Step 2: Write the model-selection comparison for the app team**

Print this table (fill in the actual `macro_f1`/delta values read in Task 3, Step 2) so whoever runs the app-side benchmark task has the desktop-side half of the decision already in hand:

| Model | Int8 TFLite size | Test macro-F1 (Int8, full test set) | Δ vs PyTorch FP32 | Acuerdo vs PyTorch |
|---|---|---|---|---|
| `efficientnet_b0` | 4.44 MB | 0.9241 | **-0.0186** (supera el umbral de 0.01) | 97.61 % |
| `efficientnet_lite0` | 3.57 MB | **0.9477** | +0.0011 | 99.58 % |
| `shufflenet_v2_x1_0` | 1.43 MB | 0.9229 | -0.0008 | 99.70 % |

Evaluacion sobre las 5 015 imagenes del split de test (`eval_tflite_int8.json` por run). Los tres siguen ≥ 0.85 de macro-F1 y ≤ 20 MB, asi que los tres son candidatos validos para el benchmark on-device; la latencia sigue sin medirse aqui.

Lectura para la seleccion: `efficientnet_lite0` es el unico que no pierde macro-F1 al cuantizar (+0.0011) y es el mas fiel al modelo FP32 (99.58 % de acuerdo), a 3.57 MB. `shufflenet_v2_x1_0` es 2.5x mas chico con practicamente la misma macro-F1 (0.9229 vs 0.9241) y el acuerdo mas alto (99.70 %), asi que es el candidato fuerte si la latencia manda. `efficientnet_b0` es el peor de los tres en esta etapa: pesa mas que `lite0`, rinde menos, y es el unico que degrada de forma apreciable al cuantizar.

Clases minoritarias a vigilar en el benchmark: `potassium_deficiency` (n=93) y `nitrogen_deficiency` (n=127) son las de peor F1 en los tres modelos. En `efficientnet_b0` caen a 0.7636 y 0.8365 respectivamente — ahi se concentra su perdida de macro-F1.

> **No leer `by_environment.lab.macro_f1` de `eval_tflite_int8.json` como calidad del modelo.** El split `lab` solo contiene 3 de las 9 clases (`common_rust` n=322, `northern_corn_leaf_blight` n=133, `gray_leaf_spot` n=77), pero la macro-F1 se promedia sobre las 9, asi que las 6 ausentes entran como F1=0 y hunden el promedio. Por eso `efficientnet_b0` aparece con `lab.macro_f1` 0.5625: recalculada solo sobre las 3 clases presentes es **0.9376**, la mejor de las tres (lite0 0.9350, shufflenet 0.9061). Para comparar por entorno usar `accuracy` (0.9624 / 0.9643 / 0.9492), que no sufre esta distorsion.

This does not get committed as a new doc file — hand it to whoever executes the app-side plan's benchmark task (Task 5 there) as the missing half of its acceptance criteria table.

- [x] **Step 3: Do NOT commit `assets/model/` yet**

The app repo's own Task 1 (companion plan) fixes that repo's `.gitignore`/Git LFS setup before anything under `assets/model/` is committed — this task only stages files on disk so that work can proceed. Confirm you're on the app repo's working tree and these files show as untracked (`git -C ../maize-doctor-app status --short assets/model` should show `??` lines, not `A ` or `M `), then stop.

---

## Self-Review Notes

- **Spec coverage:** Fase 8a's remaining checklist items (from `implementation-plan.md`) were: reconstruct architecture + load weights (already done — `best.pth` exists per model), confirm class↔index mapping (Task 1 + Task 2 Step 4), export to TFLite (Task 2), Int8 PTQ (Task 2 Step 2 — note: this pipeline's Int8 is dynamic per-channel, not calibration-based static PTQ as the original doc assumed; documented in Global Constraints), freeze/document preprocessing in a model card (superseded by the already-written `docs/es/deployment/react-native.md`, which is more complete than the originally-envisioned `model_card.json` and is not duplicated here), validate equivalence before accepting (Task 2's parity check + Task 3's full-test-set eval), verify size on desktop (Task 2 Step 3 sizes + table in Task 4).
- **Placeholder scan:** no TBD/"add error handling"/"similar to Task N" patterns present; every step has an exact command or exact code.
- **Type consistency:** `write_labels_json` signature matches its one call site in `export.py` and its one test in Task 1; no other task calls it directly.
