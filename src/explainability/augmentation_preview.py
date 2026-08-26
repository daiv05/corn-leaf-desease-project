"""Evidencia visual de las augmentations aplicadas durante el entrenamiento.

Guarda, por corrida, un grid PNG por clase (imagen original + N variantes tras
pasar por el pipeline de augmentation que efectivamente usa esa clase) para poder
verificar visualmente qué le está pasando a las imágenes sin adivinar a partir del
código de `src/data/transforms.py`.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torchvision.transforms as T
from matplotlib import pyplot as plt
from PIL import Image

from src.data.loader import load_and_normalize_image

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def _denormalize_to_uint8(tensor) -> np.ndarray:
    """Invierte Normalize/ToTensor para poder mostrar el tensor como imagen RGB."""
    array = tensor.numpy().transpose(1, 2, 0)
    array = array * _IMAGENET_STD + _IMAGENET_MEAN
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def _save_class_grid(
    class_name: str,
    pipeline_name: str,
    image: Image.Image,
    pipeline: T.Compose,
    num_variants: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, num_variants + 1, figsize=(3 * (num_variants + 1), 3.4))

    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    for i in range(num_variants):
        augmented = _denormalize_to_uint8(pipeline(image))
        axes[i + 1].imshow(augmented)
        axes[i + 1].set_title(f"Aug {i + 1}", fontsize=10)
        axes[i + 1].axis("off")

    fig.suptitle(f"{class_name}  (pipeline: {pipeline_name})", fontsize=12, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_augmentation_evidence(
    train_csv_path: str,
    dataset_root: Path,
    train_transform: T.Compose,
    minority_transform: T.Compose,
    output_dir: Path,
    minority_classes: set[str],
    num_variants: int = 4,
    seed: int = 42,
) -> None:
    """
    Por cada clase presente en `train_csv_path`, toma una imagen de muestra y guarda
    un grid (original + `num_variants` variantes) bajo `<output_dir>/augmentation_preview/`,
    usando el pipeline 'minority' si la clase está en `minority_classes`, o 'train' en
    caso contrario
    """
    df = pd.read_csv(train_csv_path)
    preview_dir = output_dir / "augmentation_preview"

    for class_name, group in df.groupby("label"):
        row = group.sample(n=1, random_state=seed).iloc[0]
        img_path = dataset_root / row["image_path"]

        try:
            image = load_and_normalize_image(img_path)
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning(f"Saltando evidencia de augmentation para '{class_name}': {e}")
            continue

        is_minority = class_name in minority_classes
        pipeline = minority_transform if is_minority else train_transform
        pipeline_name = "minority" if is_minority else "train"

        output_path = preview_dir / f"{class_name}.png"
        _save_class_grid(
            class_name=class_name,
            pipeline_name=pipeline_name,
            image=image,
            pipeline=pipeline,
            num_variants=num_variants,
            output_path=output_path,
        )

    logger.info(f"Evidencia de augmentation guardada en {preview_dir}")
