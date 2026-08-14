import builtins
import json
import sys
from pathlib import Path

import pytest
import torch

from src.export.common import (
    ExportDependencyError,
    ExportFormatResult,
    ExportReport,
    parse_export_formats,
    resolve_export_inputs,
    write_export_summary,
)
from src.export.parity import ParityResult


def test_parse_export_formats_csv():
    assert parse_export_formats("onnx,tflite") == ["onnx", "tflite"]


def test_parse_export_formats_espacios_y_mayusculas():
    assert parse_export_formats(" ONNX , tflite ") == ["onnx", "tflite"]


def test_parse_export_formats_vacio():
    assert parse_export_formats(None) == []
    assert parse_export_formats("") == []


def test_parse_export_formats_desconocido():
    with pytest.raises(SystemExit):
        parse_export_formats("onnx,coreml")


def test_parse_export_formats_deduplica():
    assert parse_export_formats("onnx,onnx,tflite") == ["onnx", "tflite"]


def test_resolve_export_inputs_sin_summary(tmp_path):
    with pytest.raises(SystemExit):
        resolve_export_inputs(tmp_path, "shufflenet_v2_x1_0", tmp_path / "dataset.yaml")


def test_resolve_export_inputs_lee_summary(tmp_path):
    summary = {
        "class_to_idx": {"healthy": 0, "common_rust": 1},
        "image_size": [224, 224],
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))

    class_to_idx, idx_to_class, image_size = resolve_export_inputs(
        tmp_path, "shufflenet_v2_x1_0", tmp_path / "dataset.yaml"
    )

    assert class_to_idx == {"healthy": 0, "common_rust": 1}
    assert idx_to_class == {0: "healthy", 1: "common_rust"}
    assert image_size == (224, 224)


def test_write_export_summary_crea_export_dir(tmp_path):
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
            ExportFormatResult(
                format="onnx",
                output_path=tmp_path / "export" / "model.onnx",
                succeeded=True,
                parity=parity,
            )
        ],
        library_versions={"torch": "2.12.1"},
    )

    write_export_summary(tmp_path, report)

    payload = json.loads((tmp_path / "export" / "export_summary.json").read_text())
    assert payload["model"] == "shufflenet_v2_x1_0"
    assert payload["formats"][0]["succeeded"] is True
    assert payload["formats"][0]["parity"]["passed"] is True
    assert Path(payload["formats"][0]["output_path"]) == Path("export/model.onnx")


def test_export_to_tflite_sin_dependencia_levanta_error_claro(monkeypatch):
    from src.export import tflite_export

    monkeypatch.setitem(sys.modules, "ai_edge_torch", None)
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "ai_edge_torch":
            raise ImportError("simulado")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(ExportDependencyError, match=r"pip install -e '\.\[export\]'"):
        tflite_export.export_to_tflite(
            torch.nn.Linear(1, 1), Path("unused.tflite"), (32, 32), torch.device("cpu")
        )
