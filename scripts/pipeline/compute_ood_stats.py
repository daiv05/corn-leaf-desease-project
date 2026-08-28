"""Calcula estadisticas de deteccion OOD (out-of-distribution) por distancia de Mahalanobis.

Para cada modelo, extrae el vector de features pooled (penultima capa, via
`FeatureExposedModel`) sobre el split de train y calcula el centroide por clase mas
una covarianza pooled compartida. El umbral de rechazo se calibra sobre el split de
val (no visto durante el entrenamiento) como el percentil pedido de las distancias de
Mahalanobis de cada muestra a su propio centroide de clase.

Escribe `<run_dir>/export/ood_stats.json`, consumido por la app movil junto al
`.tflite` de dos salidas (logits + features) para bloquear diagnosticos sobre
imagenes fuera de dominio.
"""

import argparse
import base64
import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.config import PROJECT_ROOT, get_output_root
from src.export.common import load_checkpoint_for_export, resolve_export_inputs
from src.export.data import build_test_loader, resolve_split_csv
from src.models import list_models
from src.models.feature_exposed import FeatureExposedModel
from src.training.common import resolve_run_dir, select_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula estadisticas de deteccion OOD (Mahalanobis) por modelo."
    )
    parser.add_argument(
        "--models", nargs="+", required=True, choices=list_models(), help="Modelos a procesar."
    )
    parser.add_argument("--run", default=None, help="run_id; por defecto usa latest.json.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Ruta explicita a un checkpoint .pth (solo valido con un unico modelo).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help="Directorio de runs del pipeline principal (default: <outputs>/main).",
    )
    parser.add_argument(
        "--splits-dir",
        default=None,
        dest="splits_dir",
        help="Directorio con train.csv/val.csv (default: el 'splits_dir' de summary.json).",
    )
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Percentil de calibracion del umbral sobre las distancias de val (default: 99).",
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "dataset.yaml"))
    return parser.parse_args()


@torch.no_grad()
def _extract_features(
    model: FeatureExposedModel, loader: torch.utils.data.DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """
    Corre el modelo sobre todo el loader y acumula features pooled + labels.

    @param {FeatureExposedModel} model Modelo envuelto, en eval().
    @param {DataLoader} loader Loader determinista (pipeline 'test', sin augmentation).
    @param {torch.device} device Dispositivo de inferencia.
    @returns {tuple[np.ndarray, np.ndarray]} Features (N, feature_dim) y labels (N,).
    """
    model.eval()
    all_features = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        _, features = model(images)
        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


def _compute_class_means(features: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Calcula el centroide (media) de features por clase.

    @param {np.ndarray} features Features (N, feature_dim).
    @param {np.ndarray} labels Indice de clase por muestra (N,).
    @param {int} num_classes Numero total de clases.
    @returns {np.ndarray} Centroides (num_classes, feature_dim).
    """
    feature_dim = features.shape[1]
    means = np.zeros((num_classes, feature_dim), dtype=np.float64)
    for class_idx in range(num_classes):
        class_features = features[labels == class_idx]
        if class_features.shape[0] == 0:
            raise SystemExit(f"La clase {class_idx} no tiene ninguna muestra en el split de train.")
        means[class_idx] = class_features.mean(axis=0)
    return means


def _compute_pooled_covariance(
    features: np.ndarray, labels: np.ndarray, means: np.ndarray
) -> np.ndarray:
    """
    Calcula la covarianza pooled: cada muestra se centra con el centroide de SU clase,
    luego se calcula una unica covarianza sobre todas las muestras centradas.

    Estandar del paper de Lee et al. 2018 (deteccion OOD via Mahalanobis) - una sola
    matriz compartida entre clases, mas robusta con datasets de tamano moderado que
    invertir una covarianza por clase.

    @param {np.ndarray} features Features (N, feature_dim).
    @param {np.ndarray} labels Indice de clase por muestra (N,).
    @param {np.ndarray} means Centroides por clase (num_classes, feature_dim).
    @returns {np.ndarray} Covarianza pooled (feature_dim, feature_dim).
    """
    centered = features - means[labels]
    n_samples = centered.shape[0]
    return (centered.T @ centered) / n_samples


def _encode_float32_base64(array: np.ndarray) -> str:
    """
    Codifica un array como bytes float32 crudos (row-major) en base64.

    Evita serializar millones de numeros como texto JSON (un array 1024x1024
    en JSON de texto pesa ~20MB; el binario float32 equivalente son 4MB, ~5.3MB
    en base64). La app decodifica el string y reinterpreta los bytes como
    Float32Array, sin parsear un array JSON gigante.

    @param {np.ndarray} array Array numerico de cualquier forma.
    @returns {str} Bytes float32 (row-major) codificados en base64.
    """
    return base64.b64encode(array.astype(np.float32).tobytes()).decode("ascii")


def _mahalanobis_distances(
    features: np.ndarray, labels: np.ndarray, means: np.ndarray, inv_covariance: np.ndarray
) -> np.ndarray:
    """
    Distancia de Mahalanobis de cada muestra a el centroide de SU propia clase.

    @param {np.ndarray} features Features (N, feature_dim).
    @param {np.ndarray} labels Indice de clase por muestra (N,).
    @param {np.ndarray} means Centroides por clase (num_classes, feature_dim).
    @param {np.ndarray} inv_covariance Inversa (pseudo-inversa) de la covarianza pooled.
    @returns {np.ndarray} Distancias (N,).
    """
    diff = features - means[labels]
    return np.einsum("ij,jk,ik->i", diff, inv_covariance, diff)


def _compute_one(args: argparse.Namespace, model_name: str, output_dir: Path) -> None:
    config_path = Path(args.config)

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        run_dir = checkpoint_path.parent
    else:
        run_dir = resolve_run_dir(output_dir, model_name, args.run)
        checkpoint_path = run_dir / "best.pth"

    class_to_idx, idx_to_class, image_size = resolve_export_inputs(run_dir, model_name, config_path)
    num_classes = len(class_to_idx)
    device = select_device()

    base_model = load_checkpoint_for_export(checkpoint_path, model_name, class_to_idx, device)
    model = FeatureExposedModel(base_model, model_name).to(device)
    model.eval()

    train_csv = resolve_split_csv(run_dir, args.splits_dir, "train")
    val_csv = resolve_split_csv(run_dir, args.splits_dir, "val")

    logger.info("Extrayendo features de train (%s)...", train_csv)
    train_loader, _ = build_test_loader(
        train_csv, config_path, class_to_idx, image_size, args.batch_size
    )
    train_features, train_labels = _extract_features(model, train_loader, device)
    logger.info("Features de train: %s", train_features.shape)

    means = _compute_class_means(train_features, train_labels, num_classes)
    covariance = _compute_pooled_covariance(train_features, train_labels, means)
    inv_covariance = np.linalg.pinv(covariance)

    logger.info("Extrayendo features de val (%s) para calibrar el umbral...", val_csv)
    val_loader, _ = build_test_loader(val_csv, config_path, class_to_idx, image_size, args.batch_size)
    val_features, val_labels = _extract_features(model, val_loader, device)

    val_distances = _mahalanobis_distances(val_features, val_labels, means, inv_covariance)
    threshold = float(np.percentile(val_distances, args.percentile))

    if not np.isfinite(threshold) or threshold <= 0:
        raise SystemExit(
            f"Umbral calibrado invalido ({threshold}) para '{model_name}'. "
            "Revisa que la covarianza pooled no este degenerada."
        )

    export_dir = run_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    # mean_per_class/inv_covariance van en base64 (float32 binario), no como arrays
    # JSON de texto: un array 1024x1024 en texto pesa ~20MB; el binario equivalente
    # son ~5.3MB en base64. Ver `_encode_float32_base64`.
    feature_dim = int(train_features.shape[1])
    payload = {
        "schema_version": 2,
        "model": model_name,
        "num_classes": num_classes,
        "feature_dim": feature_dim,
        "mean_per_class_b64": _encode_float32_base64(means),
        "inv_covariance_b64": _encode_float32_base64(inv_covariance),
        "threshold": round(threshold, 6),
        "percentile": args.percentile,
        "calibration_split": "val",
        "labels": [idx_to_class[i] for i in range(num_classes)],
    }
    output_path = export_dir / "ood_stats.json"
    output_path.write_text(json.dumps(payload))

    logger.info(
        "OK: %s -> %s (feature_dim=%d, threshold=%.4f, percentile=%.1f)",
        model_name,
        output_path,
        train_features.shape[1],
        threshold,
        args.percentile,
    )


def main() -> None:
    args = _parse_args()
    if args.checkpoint and len(args.models) > 1:
        raise SystemExit(
            "--checkpoint apunta a un unico archivo; no se puede usar con varios --models."
        )

    output_root = get_output_root()
    output_dir = Path(args.output_dir) if args.output_dir else output_root / "main"

    for model_name in args.models:
        _compute_one(args, model_name, output_dir)


if __name__ == "__main__":
    main()
