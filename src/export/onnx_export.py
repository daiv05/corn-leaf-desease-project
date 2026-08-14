from __future__ import annotations

from pathlib import Path

import torch


def export_to_onnx(
    model: torch.nn.Module,
    output_path: Path,
    image_size: tuple[int, int],
    device: torch.device,
    opset: int = 18,
) -> Path:
    """
    Exporta `model` a ONNX con batch dinámico y alto/ancho estáticos.

    @param {torch.nn.Module} model Modelo en modo eval().
    @param {Path} output_path Ruta destino del archivo .onnx.
    @param {tuple[int, int]} image_size Alto y ancho de entrada (h, w).
    @param {torch.device} device Dispositivo donde correr el tracing.
    @param {int} opset Versión de opset ONNX a usar.
    @returns {Path} `output_path`.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    dummy_input = torch.randn(1, 3, *image_size, device=device)
    batch_size = torch.export.Dim("batch_size")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes=({0: batch_size},),
        opset_version=opset,
        verbose=False,
    )
    return output_path
