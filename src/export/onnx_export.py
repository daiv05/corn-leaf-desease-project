from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# Limite de protobuf de ONNX (2 GB). Por debajo se guarda todo en un solo archivo;
# los modelos del pipeline principal pesan ~5-16 MB, asi que siempre caen aqui.
_SINGLE_FILE_LIMIT_BYTES = 1_800_000_000


def _consolidate_single_file(model_path: Path) -> None:
    """
    Reescribe el .onnx con los pesos embebidos y borra el sidecar .onnx.data.

    El exportador de torch saca los initializers a un archivo externo `model.onnx.data`.
    Para la app movil eso significa empaquetar y mantener sincronizados dos archivos, y
    un `.onnx` suelto sin su `.data` carga pero falla al inferir. Se consolida siempre
    que quepa en el limite de protobuf.

    @param {Path} model_path Ruta al .onnx exportado.
    """
    import onnx

    external_data = model_path.with_suffix(model_path.suffix + ".data")
    if not external_data.exists():
        return
    if external_data.stat().st_size > _SINGLE_FILE_LIMIT_BYTES:
        logger.warning(
            "El modelo supera el limite de archivo unico de ONNX; se conserva %s.",
            external_data.name,
        )
        return

    model = onnx.load(str(model_path))  # resuelve los pesos externos
    onnx.save(model, str(model_path), save_as_external_data=False)
    external_data.unlink()


def _quantize_onnx_dynamic(model_path: Path) -> None:
    """
    Cuantiza un .onnx a int8 dinamico in-place (pesos int8, activaciones en runtime).

    Antes de cuantizar se borra el `value_info` intermedio: el exportador dynamo de torch
    deja anotaciones de shape inconsistentes con el grafo real (reporta la dimension de
    features donde va la de clases), y la inferencia de shapes que corre onnxruntime al
    cuantizar aborta por esa discrepancia. El grafo en si es correcto - la variante FP32
    da paridad exacta -, asi que basta con dejar que las shapes se re-infieran.

    @param {Path} model_path Modelo a cuantizar, sobrescrito con la version int8.
    @throws {ExportDependencyError} Si falta onnxruntime.
    """
    try:
        import onnx
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as e:
        from src.export.common import ExportDependencyError

        raise ExportDependencyError(
            "Cuantizacion int8 de ONNX requiere 'onnx' y 'onnxruntime'. "
            "Instala con: pip install -e '.[export]'"
        ) from e

    model = onnx.load(str(model_path))
    del model.graph.value_info[:]
    onnx.save(model, str(model_path), save_as_external_data=False)

    quantize_dynamic(
        model_input=str(model_path),
        model_output=str(model_path),
        weight_type=QuantType.QInt8,
    )


def export_to_onnx(
    model: torch.nn.Module,
    output_path: Path,
    image_size: tuple[int, int],
    device: torch.device,
    opset: int = 18,
    quantize: str | None = None,
) -> Path:
    """
    Exporta `model` a ONNX con batch dinámico y alto/ancho estáticos, en un solo archivo.

    @param {torch.nn.Module} model Modelo en modo eval().
    @param {Path} output_path Ruta destino del archivo .onnx.
    @param {tuple[int, int]} image_size Alto y ancho de entrada (h, w).
    @param {torch.device} device Dispositivo donde correr el tracing.
    @param {int} opset Versión de opset ONNX a usar.
    @param {str|None} quantize "int8" para cuantización dinámica; None/"none" para FP32.
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

    _consolidate_single_file(output_path)

    if quantize == "int8":
        _quantize_onnx_dynamic(output_path)

    return output_path
