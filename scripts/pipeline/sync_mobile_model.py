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
