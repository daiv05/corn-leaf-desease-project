import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, f1_score
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
_DEFAULT_OUTPUT_DIR = str(DATASET_ROOT / "results" / "baselines")


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


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.inference_mode()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        preds = model(images.to(device)).argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    return all_preds, all_labels


def train_baseline(
    model_name: str,
    splits_dir: Path,
    output_dir: Path,
    num_classes: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
    idx_to_class: dict[int, str],
    seed: int,
) -> None:
    logger.info(f"[{model_name}] Iniciando entrenamiento")

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=lambda wid: _worker_init_fn(wid, seed=seed),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    model = MODEL_REGISTRY.build(model_name, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_ckpt_path = output_dir / model_name / "best.pth"
    best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_preds, val_labels = _evaluate(model, val_loader, device)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        logger.info(
            f"[{model_name}] Epoch {epoch}/{epochs} - "
            f"loss: {train_loss:.4f} | val macro-F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_ckpt_path)

    # Evaluación final sobre test con el mejor checkpoint
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_preds, test_labels = _evaluate(model, test_loader, device)

    target_names = [idx_to_class[i] for i in range(num_classes)]
    test_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    test_acc = sum(p == gt for p, gt in zip(test_preds, test_labels)) / len(test_labels)
    report = classification_report(
        test_labels, test_preds, target_names=target_names, zero_division=0
    )

    metrics = {
        "model": model_name,
        "epochs": epochs,
        "best_val_macro_f1": round(best_val_f1, 6),
        "test_accuracy": round(test_acc, 6),
        "test_macro_f1": round(test_f1, 6),
        "classification_report": report,
    }

    metrics_path = output_dir / model_name / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    logger.info(f"[{model_name}] Métricas guardadas en {metrics_path}")
    logger.info(f"[{model_name}] test accuracy={test_acc:.4f} | test macro-F1={test_f1:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena modelos baseline de Deep Learning.")
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
            "Genera los splits primero con: make splits  (o make splits-sample para muestra)"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo: {device}")

    train_df = pd.read_csv(splits_dir / "train.csv")
    present_classes = sorted(train_df["label"].unique())
    allowed = [c for c in cfg["dataset"]["classes"] if c in present_classes]
    class_to_idx = {name: idx for idx, name in enumerate(allowed)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    num_classes = len(allowed)

    logger.info(f"Modelos a entrenar: {model_names}")
    logger.info(f"Splits: {splits_dir}  |  Clases: {num_classes}  |  Epochs: {args.epochs}")

    for model_name in model_names:
        train_baseline(
            model_name=model_name,
            splits_dir=splits_dir,
            output_dir=output_dir,
            num_classes=num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            idx_to_class=idx_to_class,
            seed=seed,
        )

    logger.info("Entrenamiento de baselines completado.")


if __name__ == "__main__":
    main()
