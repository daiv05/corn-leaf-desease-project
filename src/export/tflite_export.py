from __future__ import annotations

from pathlib import Path

import torch


def export_to_tflite(
    model: torch.nn.Module,
    output_path: Path,
    image_size: tuple[int, int],
    device: torch.device,
) -> Path:
    """
    Exporta `model` a TFLite con batch fijo=1, vía ai-edge-torch.

    @param {torch.nn.Module} model Modelo en modo eval().
    @param {Path} output_path Ruta destino del archivo .tflite.
    @param {tuple[int, int]} image_size Alto y ancho de entrada (h, w).
    @param {torch.device} device Dispositivo donde correr el tracing.
    @returns {Path} `output_path`.
    @throws {ExportDependencyError} Si ai-edge-torch no está instalado.
    """
    try:
        import ai_edge_torch
    except ImportError as e:
        from src.export.common import ExportDependencyError

        raise ExportDependencyError(
            "TFLite export requiere el extra 'export': pip install -e '.[export]'"
        ) from e

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    dummy_input = torch.randn(1, 3, *image_size, device=device)
    edge_model = ai_edge_torch.convert(model, (dummy_input,))
    edge_model.export(str(output_path))
    return output_path
