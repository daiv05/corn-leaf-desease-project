import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
import yaml

from src.config import PROJECT_ROOT, get_output_root
from src.data.dataset import resolve_class_mapping
from src.data.loader import load_and_normalize_image
from src.data.transforms import CornTransformFactory
from src.models import build_model, list_models
from src.training.common import resolve_run_dir, select_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_IMAGE_SIZE_BY_MODEL = {
    "mobilenet_v3_large": 224,
    "mobilenet_v3_small": 224,
    "efficientnet_b4": 380,
}


def _load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _default_splits_dir(output_root: Path, baseline: bool, full: bool) -> Path:
    if baseline and full:
        raise SystemExit("Usa solo uno de --baseline o --full.")
    if baseline:
        return output_root / "splits" / "seed_42_baseline"
    if full:
        return output_root / "splits" / "seed_42"

    baseline_dir = output_root / "splits" / "seed_42_baseline"
    return baseline_dir if baseline_dir.exists() else output_root / "splits" / "seed_42"


def _load_summary(checkpoint_path: Path) -> dict[str, Any]:
    summary_path = checkpoint_path.parent / "summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, "r") as f:
        return json.load(f)


def _resolve_class_mapping(
    summary: dict[str, Any],
    splits_dir: Path,
    cfg: dict[str, Any],
) -> tuple[dict[str, int], dict[int, str]]:
    if "class_to_idx" in summary:
        class_to_idx = {str(name): int(idx) for name, idx in summary["class_to_idx"].items()}
        idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        return class_to_idx, idx_to_class

    train_csv = splits_dir / "train.csv"
    if not train_csv.exists():
        raise SystemExit(
            f"No existe {train_csv}. Pasa --splits-dir o conserva summary.json junto al checkpoint."
        )
    return resolve_class_mapping(str(train_csv), cfg["dataset"]["classes"])


def _resolve_target_size(
    args: argparse.Namespace,
    summary: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[int, int]:
    if args.image_size is not None:
        return (args.image_size, args.image_size)

    image_size = summary.get("image_size")
    if isinstance(image_size, list) and len(image_size) == 2:
        return (int(image_size[0]), int(image_size[1]))

    default_size = _DEFAULT_IMAGE_SIZE_BY_MODEL.get(args.model)
    if default_size is not None:
        return (default_size, default_size)

    height, width = cfg["dataset"]["target_size"]
    return (height, width)


def _load_state_dict(checkpoint_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"Checkpoint invalido: {checkpoint_path}")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Predice la clase de una imagen de hoja de maiz.")
    parser.add_argument(
        "--model",
        required=True,
        choices=list_models(),
        help="Nombre interno del modelo registrado.",
    )
    parser.add_argument("--image", required=True, help="Ruta a la imagen a clasificar.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Ruta al checkpoint .pth (default: último run en <repo>/outputs/baselines).",
    )
    parser.add_argument(
        "--run",
        default=None,
        help="run_id específico a usar. Por defecto usa latest.json para el modelo.",
    )
    parser.add_argument(
        "--splits-dir",
        default=None,
        dest="splits_dir",
        help="Directorio con train.csv para reconstruir class_to_idx si no hay summary.json.",
    )
    parser.add_argument("--baseline", action="store_true", help="Usa splits/seed_42_baseline.")
    parser.add_argument("--full", action="store_true", help="Usa splits/seed_42.")
    parser.add_argument("--image-size", type=int, default=None, dest="image_size")
    parser.add_argument("--top-k", type=int, default=3, dest="top_k")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "dataset.yaml"),
    )
    args = parser.parse_args()

    output_root = get_output_root()
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = (
            resolve_run_dir(
                output_root / "baselines",
                args.model,
                args.run,
            )
            / "best.pth"
        )
    if not checkpoint_path.exists():
        raise SystemExit(f"No existe el checkpoint: {checkpoint_path}")

    config_path = Path(args.config)
    cfg = _load_config(config_path)
    summary = _load_summary(checkpoint_path)
    splits_dir = (
        Path(args.splits_dir)
        if args.splits_dir
        else _default_splits_dir(output_root, baseline=args.baseline, full=args.full)
    )
    class_to_idx, idx_to_class = _resolve_class_mapping(summary, splits_dir, cfg)
    target_size = _resolve_target_size(args, summary, cfg)

    device = select_device()
    model = build_model(args.model, num_classes=len(class_to_idx), pretrained=False).to(device)
    model.load_state_dict(_load_state_dict(checkpoint_path, device))
    model.eval()

    factory = CornTransformFactory(config_path=str(config_path), target_size=target_size)
    image = load_and_normalize_image(args.image)
    tensor = factory.get_pipeline("inference")(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

    top_k = min(args.top_k, len(idx_to_class))
    values, indices = torch.topk(probabilities, k=top_k)
    prediction = idx_to_class[int(indices[0])]

    print(f"Modelo: {args.model}")
    print(f"Imagen: {args.image}")
    print(f"Prediccion: {prediction} ({float(values[0]):.4f})")
    print("Top-k:")
    for prob, idx in zip(values.tolist(), indices.tolist(), strict=True):
        print(f"  {idx_to_class[int(idx)]}: {prob:.4f}")


if __name__ == "__main__":
    main()
