from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

_MIN_RELIABLE_SAMPLES = 20


def _extract_logits(model_output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    """
    Extrae el tensor de logits de la salida de un modelo, sea de un solo output
    (Tensor) o de un `FeatureExposedModel` (tupla `(logits, features)`).

    @param {torch.Tensor|tuple[torch.Tensor, ...]} model_output Salida de `model(images)`.
    @returns {torch.Tensor} El tensor de logits (primer elemento si es tupla).
    """
    return model_output[0] if isinstance(model_output, tuple) else model_output


@dataclass
class ParityResult:
    format: str
    n_samples: int
    torch_top1_accuracy: float
    exported_top1_accuracy: float
    agreement_rate: float
    max_abs_prob_diff: float
    mean_abs_prob_diff: float
    tolerance: float
    passed: bool
    min_agreement_rate: float = 1.0
    warnings: list[str] = field(default_factory=list)


def _collect_samples(
    test_loader: DataLoader, device: torch.device, sample_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Toma las primeras `sample_size` muestras de `test_loader` (pipeline 'test',
    determinista, sin augmentation).

    @param {DataLoader} test_loader Loader de test ya construido.
    @param {torch.device} device Dispositivo para las imágenes recolectadas.
    @param {int} sample_size Número máximo de muestras a recolectar.
    @returns {tuple[torch.Tensor, torch.Tensor]} Imágenes apiladas y sus labels.
    """
    images_batches: list[torch.Tensor] = []
    labels_batches: list[torch.Tensor] = []
    collected = 0
    for images, labels in test_loader:
        images_batches.append(images)
        labels_batches.append(labels)
        collected += images.size(0)
        if collected >= sample_size:
            break
    images = torch.cat(images_batches, dim=0)[:sample_size].to(device)
    labels = torch.cat(labels_batches, dim=0)[:sample_size]
    return images, labels


def _build_parity_result(
    format_name: str,
    torch_probs: np.ndarray,
    exported_probs: np.ndarray,
    labels: np.ndarray,
    tolerance: float,
    min_agreement_rate: float = 1.0,
) -> ParityResult:
    torch_preds = torch_probs.argmax(axis=1)
    exported_preds = exported_probs.argmax(axis=1)

    torch_top1_accuracy = float((torch_preds == labels).mean())
    exported_top1_accuracy = float((exported_preds == labels).mean())
    agreement_rate = float((torch_preds == exported_preds).mean())

    abs_diff = np.abs(torch_probs - exported_probs)
    max_abs_prob_diff = float(abs_diff.max())
    mean_abs_prob_diff = float(abs_diff.mean())

    warnings: list[str] = []
    n_samples = len(labels)
    if n_samples < _MIN_RELIABLE_SAMPLES:
        warnings.append(
            f"n_samples={n_samples} < {_MIN_RELIABLE_SAMPLES}, resultado poco confiable"
        )

    passed = max_abs_prob_diff <= tolerance and agreement_rate >= min_agreement_rate

    return ParityResult(
        format=format_name,
        n_samples=n_samples,
        torch_top1_accuracy=torch_top1_accuracy,
        exported_top1_accuracy=exported_top1_accuracy,
        agreement_rate=agreement_rate,
        max_abs_prob_diff=max_abs_prob_diff,
        mean_abs_prob_diff=mean_abs_prob_diff,
        tolerance=tolerance,
        passed=passed,
        min_agreement_rate=min_agreement_rate,
        warnings=warnings,
    )


def validate_onnx_parity(
    torch_model: torch.nn.Module,
    onnx_path: Path,
    test_loader: DataLoader,
    device: torch.device,
    sample_size: int = 30,
    tolerance: float = 1e-3,
    min_agreement_rate: float = 1.0,
) -> ParityResult:
    """
    Compara probabilidades del modelo PyTorch contra el modelo ONNX exportado.

    @param {torch.nn.Module} torch_model Modelo original en modo eval().
    @param {Path} onnx_path Ruta al modelo .onnx exportado.
    @param {DataLoader} test_loader Loader del split de test (pipeline 'test').
    @param {torch.device} device Dispositivo para correr el modelo PyTorch.
    @param {int} sample_size Número de muestras a comparar.
    @param {float} tolerance Tolerancia máxima de diferencia absoluta de probabilidad.
    @param {float} min_agreement_rate Acuerdo top-1 mínimo exigido (1.0 = exacto).
    @returns {ParityResult} Resultado de la comparación.
    @throws {ExportDependencyError} Si onnxruntime no está instalado.
    """
    try:
        import onnxruntime as ort
    except ImportError as e:
        from src.export.common import ExportDependencyError

        raise ExportDependencyError(
            "Validacion de paridad ONNX requiere 'onnxruntime'. "
            "Instala con: pip install -e '.[export]'"
        ) from e

    images, labels = _collect_samples(test_loader, device, sample_size)

    torch_model.eval()
    with torch.no_grad():
        torch_logits = _extract_logits(torch_model(images))
        torch_probs = torch_logits.softmax(dim=1).cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_logits = session.run(None, {input_name: images.cpu().numpy()})[0]
    exported_probs = torch.from_numpy(onnx_logits).softmax(dim=1).numpy()

    return _build_parity_result(
        "onnx", torch_probs, exported_probs, labels.numpy(), tolerance, min_agreement_rate
    )


def validate_tflite_parity(
    torch_model: torch.nn.Module,
    tflite_path: Path,
    test_loader: DataLoader,
    device: torch.device,
    sample_size: int = 30,
    tolerance: float = 1e-3,
    min_agreement_rate: float = 1.0,
) -> ParityResult:
    """
    Compara probabilidades del modelo PyTorch contra el modelo TFLite exportado.

    @param {torch.nn.Module} torch_model Modelo original en modo eval().
    @param {Path} tflite_path Ruta al modelo .tflite exportado.
    @param {DataLoader} test_loader Loader del split de test (pipeline 'test').
    @param {torch.device} device Dispositivo para correr el modelo PyTorch.
    @param {int} sample_size Número de muestras a comparar.
    @param {float} tolerance Tolerancia máxima de diferencia absoluta de probabilidad.
    @param {float} min_agreement_rate Acuerdo top-1 mínimo exigido (1.0 = exacto).
    @returns {ParityResult} Resultado de la comparación.
    @throws {ExportDependencyError} Si no hay un intérprete TFLite disponible.
    """
    interpreter_cls = None
    try:
        from ai_edge_litert.interpreter import Interpreter as interpreter_cls
    except ImportError:
        try:
            from tensorflow.lite import Interpreter as interpreter_cls
        except ImportError:
            pass

    if interpreter_cls is None:
        from src.export.common import ExportDependencyError

        raise ExportDependencyError(
            "Validacion de paridad TFLite requiere 'ai-edge-litert' o 'tensorflow'. "
            "Instala con: pip install -e '.[export]'"
        )

    images, labels = _collect_samples(test_loader, device, sample_size)

    torch_model.eval()
    with torch.no_grad():
        torch_logits = _extract_logits(torch_model(images))
        torch_probs = torch_logits.softmax(dim=1).cpu().numpy()

    interpreter = interpreter_cls(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    exported_logits = np.zeros(
        (images.size(0), output_detail["shape"][-1]), dtype=np.float32
    )
    images_np = images.cpu().numpy()
    for i in range(images_np.shape[0]):
        interpreter.set_tensor(input_detail["index"], images_np[i : i + 1])
        interpreter.invoke()
        exported_logits[i] = interpreter.get_tensor(output_detail["index"])[0]

    exported_probs = torch.from_numpy(exported_logits).softmax(dim=1).numpy()

    return _build_parity_result(
        "tflite", torch_probs, exported_probs, labels.numpy(), tolerance, min_agreement_rate
    )
