"""Valida el detector OOD (Mahalanobis) calibrado en compute_ood_stats.py.

Corre el detector sobre (a) una muestra de imagenes de test legitimas -esperado:
por debajo del umbral- y (b) imagenes fuera de dominio -esperado: por encima del
umbral- (fotos negras/ruido/color solido, sustitutas de fotos de escritorio/objetos
random). Reporta la tasa de falsos positivos sobre datos legitimos y si las OOD
sinteticas quedan claramente separadas.

Uso: python scripts/checks/validate_ood_detector.py --model shufflenet_v2_x1_0
     --checkpoint <run_dir>/best.pth --ood-stats <run_dir>/export/ood_stats.json
"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch

from src.config import PROJECT_ROOT
from src.data.transforms import CornTransformFactory
from src.export.common import load_checkpoint_for_export, resolve_export_inputs
from src.models.feature_exposed import FeatureExposedModel
from src.training.common import resolve_run_dir, select_device


def _load_ood_stats(path: Path) -> dict:
    data = json.loads(path.read_text())
    num_classes = data["num_classes"]
    feature_dim = data["feature_dim"]
    means = np.frombuffer(
        base64.b64decode(data["mean_per_class_b64"]), dtype=np.float32
    ).reshape(num_classes, feature_dim)
    inv_covariance = np.frombuffer(
        base64.b64decode(data["inv_covariance_b64"]), dtype=np.float32
    ).reshape(feature_dim, feature_dim)
    return {
        "means": means,
        "inv_covariance": inv_covariance,
        "threshold": data["threshold"],
        "labels": data["labels"],
    }


def _mahalanobis_min_distance(feature: np.ndarray, means: np.ndarray, inv_covariance: np.ndarray) -> float:
    diffs = feature[None, :] - means
    distances = np.einsum("ij,jk,ik->i", diffs, inv_covariance, diffs)
    return float(distances.min())


def _synthetic_ood_images(image_size: tuple[int, int]) -> dict[str, torch.Tensor]:
    """Genera tensores sinteticos fuera de dominio (mismo preprocess que una foto real)."""
    h, w = image_size
    factory = CornTransformFactory(target_size=image_size)
    transform = factory.get_pipeline("test")

    from PIL import Image

    rng = np.random.default_rng(42)
    samples = {
        "black": np.zeros((h, w, 3), dtype=np.uint8),
        "white": np.full((h, w, 3), 255, dtype=np.uint8),
        "gray_solid": np.full((h, w, 3), 128, dtype=np.uint8),
        "random_noise": rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8),
        "blue_solid": np.tile(np.array([30, 60, 200], dtype=np.uint8), (h, w, 1)),
    }
    return {name: transform(Image.fromarray(arr)) for name, arr in samples.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ood-stats", required=True, dest="ood_stats")
    parser.add_argument("--run", default=None)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "dataset.yaml"))
    parser.add_argument(
        "--n-legit-samples",
        type=int,
        default=30,
        help="Cuantas imagenes legitimas de test evaluar (requiere DATASET_ROOT accesible).",
    )
    parser.add_argument(
        "--splits-dir",
        default=None,
        dest="splits_dir",
        help="Directorio con test.csv (default: el 'splits_dir' de summary.json).",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    run_dir = checkpoint_path.parent
    config_path = Path(args.config)

    class_to_idx, idx_to_class, image_size = resolve_export_inputs(run_dir, args.model, config_path)
    device = select_device()
    base_model = load_checkpoint_for_export(checkpoint_path, args.model, class_to_idx, device)
    model = FeatureExposedModel(base_model, args.model).to(device)
    model.eval()

    stats = _load_ood_stats(Path(args.ood_stats))
    print(f"threshold={stats['threshold']:.4f}\n")

    print("=== Imagenes sinteticas fuera de dominio (esperado: distancia > threshold) ===")
    synthetic = _synthetic_ood_images(image_size)
    ood_flagged = 0
    for name, tensor in synthetic.items():
        with torch.no_grad():
            _, features = model(tensor.unsqueeze(0).to(device))
        distance = _mahalanobis_min_distance(
            features.cpu().numpy()[0], stats["means"], stats["inv_covariance"]
        )
        flagged = distance > stats["threshold"]
        ood_flagged += int(flagged)
        status = "OOD (correcto)" if flagged else "NO detectado (FALLO)"
        print(f"  {name:15s} distance={distance:12.2f}  {status}")

    print(f"\nOOD sinteticas detectadas: {ood_flagged}/{len(synthetic)}")

    if args.n_legit_samples > 0:
        print("\n=== Muestra de imagenes legitimas de test (esperado: distancia <= threshold) ===")
        from src.export.data import build_test_loader, resolve_test_csv

        test_csv = resolve_test_csv(run_dir, args.splits_dir)
        loader, _ = build_test_loader(
            test_csv, config_path, class_to_idx, image_size, batch_size=1
        )
        evaluated = 0
        false_positives = 0
        for images, labels in loader:
            if evaluated >= args.n_legit_samples:
                break
            with torch.no_grad():
                _, features = model(images.to(device))
            distance = _mahalanobis_min_distance(
                features.cpu().numpy()[0], stats["means"], stats["inv_covariance"]
            )
            flagged = distance > stats["threshold"]
            false_positives += int(flagged)
            evaluated += 1

        if evaluated == 0:
            print("  No se pudo leer ninguna imagen legitima (DATASET_ROOT inaccesible?).")
        else:
            fp_rate = false_positives / evaluated
            print(
                f"  Evaluadas: {evaluated}, falsos positivos: {false_positives} "
                f"({fp_rate:.1%})"
            )


if __name__ == "__main__":
    main()
