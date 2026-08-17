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
