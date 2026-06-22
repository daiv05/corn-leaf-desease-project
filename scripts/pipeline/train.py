import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

import src.models.baselines.efficientnet  # noqa: F401 - registra modelos
import src.models.baselines.mobilenet  # noqa: F401 - registra modelos
from src.config import DATASET_ROOT, PROJECT_ROOT, set_global_seed
from src.data.dataset import CornDataset, build_weighted_sampler
from src.data.transforms import CornTransformFactory
from src.models.registry import MODEL_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_SPLITS_DIR = str(DATASET_ROOT / "splits" / "seed_42")
_DEFAULT_OUTPUT_DIR = str(DATASET_ROOT / "results" / "main")


def _resolve_model_names(requested: list[str]) -> list[str]:
    available = MODEL_REGISTRY.list_names()
    if requested == ["all"]:
        return available
    unknown = [n for n in requested if n not in MODEL_REGISTRY]
    if unknown:
        raise SystemExit(f"Modelos desconocidos: {unknown}. Disponibles: {available}")
    return requested


def _worker_init_fn(worker_id: int, seed: int) -> None:
    import random

    import numpy as np

    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_class_weights(dataset: CornDataset) -> torch.Tensor:
    counts = dataset.data_frame["label"].value_counts().to_dict()
    total = sum(counts.values())
    num_classes = len(dataset.class_to_idx)
    weights = [total / (num_classes * counts[cls]) for cls in dataset.allowed_classes]
    return torch.tensor(weights, dtype=torch.float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el pipeline principal de Deep Learning.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help='Nombres de modelos a entrenar, o "all" para todos. '
        f"Disponibles: {MODEL_REGISTRY.list_names()}",
    )
    parser.add_argument(
        "--splits-dir",
        default=_DEFAULT_SPLITS_DIR,
        dest="splits_dir",
        help=f"Directorio con train/val/test.csv (default: {_DEFAULT_SPLITS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        dest="output_dir",
        help=f"Directorio de salida para checkpoints y métricas (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "dataset.yaml"),
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seed = cfg["dataset"]["seed"]
    set_global_seed(seed)

    model_names = _resolve_model_names(args.models)
    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)

    if not splits_dir.exists():
        raise SystemExit(
            f"El directorio de splits no existe: {splits_dir}\n"
            "Genera los splits primero con: make splits"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo: {device}")
    logger.info(f"Modelos a entrenar: {model_names}")

    factory = CornTransformFactory()
    train_dataset = CornDataset(
        csv_path=str(splits_dir / "train.csv"),
        transform=factory.get_pipeline("train"),
        minority_transform=factory.get_pipeline("minority"),
    )
    val_dataset = CornDataset(
        csv_path=str(splits_dir / "val.csv"),
        transform=factory.get_pipeline("val"),
    )
    test_dataset = CornDataset(
        csv_path=str(splits_dir / "test.csv"),
        transform=factory.get_pipeline("test"),
    )

    sampler = build_weighted_sampler(train_dataset, seed=seed)
    class_weights = build_class_weights(train_dataset).to(device)
    num_classes = len(train_dataset.class_to_idx)

    train_loader = DataLoader(  # noqa: F841
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=lambda wid: _worker_init_fn(wid, seed=seed),
    )
    val_loader = DataLoader(  # noqa: F841
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(  # noqa: F841
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)  # noqa: F841

    for model_name in model_names:
        model = MODEL_REGISTRY.build(model_name, num_classes=num_classes).to(device)  # noqa: F841
        ckpt_dir = output_dir / model_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[{model_name}] Modelo construido. Checkpoints en {ckpt_dir}")

        # TODO: loop de entrenamiento


if __name__ == "__main__":
    main()
