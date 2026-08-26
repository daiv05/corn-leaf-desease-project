"""Parseo del modo de cuantizacion y nombrado de los artefactos exportados."""

import pytest

from src.export.common import (
    _PARITY_DEFAULTS,
    export_artifact_name,
    parse_quantize,
)


def test_parse_quantize_int8():
    assert parse_quantize("int8") == "int8"
    assert parse_quantize(" INT8 ") == "int8"


def test_parse_quantize_vacio_es_fp32():
    assert parse_quantize(None) is None
    assert parse_quantize("") is None
    assert parse_quantize("none") is None
    assert parse_quantize("fp32") is None


def test_parse_quantize_desconocido():
    with pytest.raises(SystemExit):
        parse_quantize("int4")


def test_export_artifact_name_fp32_no_lleva_sufijo():
    assert export_artifact_name("onnx", None) == "model.onnx"
    assert export_artifact_name("tflite", None) == "model.tflite"


def test_export_artifact_name_cuantizado_no_pisa_al_fp32():
    assert export_artifact_name("onnx", "int8") == "model_int8.onnx"
    assert export_artifact_name("tflite", "int8") == "model_int8.tflite"


def test_umbrales_de_paridad_se_relajan_al_cuantizar():
    """Exigirle a int8 la tolerancia de FP32 lo reprobaria siempre."""
    fp32 = _PARITY_DEFAULTS[None]
    int8 = _PARITY_DEFAULTS["int8"]
    assert int8["tolerance"] > fp32["tolerance"]
    assert int8["min_agreement_rate"] < fp32["min_agreement_rate"]
    assert fp32["min_agreement_rate"] == 1.0
