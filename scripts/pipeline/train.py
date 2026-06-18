import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.config import DATASET_ROOT
from src.data.dataset import CornDataset
from src.data.transforms import CornTransformFactory

EXCLUDE_CLASSES = ["aphids_pest"]
SPLITS_DIR = DATASET_ROOT / "splits" / "seed_42"


def build_weighted_sampler(dataset: CornDataset) -> WeightedRandomSampler:
    """Construye un WeightedRandomSampler que iguala la frecuencia de aparición por clase."""
    labels = dataset.data_frame["label"].tolist()
    class_counts = dataset.data_frame["label"].value_counts().to_dict()

    # Peso por muestra = 1 / tamaño de su clase
    sample_weights = torch.tensor(
        [1.0 / class_counts[label] for label in labels], dtype=torch.float
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_class_weights(dataset: CornDataset) -> torch.Tensor:
    """
    Calcula pesos de clase para CrossEntropyLoss:
        w_i = total / (num_classes * count_i)
    Devuelve un tensor ordenado por class_to_idx.
    """
    counts = dataset.data_frame["label"].value_counts().to_dict()
    total = sum(counts.values())
    num_classes = len(dataset.class_to_idx)
    weights = [
        total / (num_classes * counts[cls])
        for cls in dataset.allowed_classes
    ]
    return torch.tensor(weights, dtype=torch.float)


def main() -> None:
    factory = CornTransformFactory()
    train_t    = factory.get_pipeline("train")
    minority_t = factory.get_pipeline("minority")
    val_t      = factory.get_pipeline("val")
    test_t     = factory.get_pipeline("test")

    train_dataset = CornDataset(
        csv_path=str(SPLITS_DIR / "train.csv"),
        transform=train_t,
        minority_transform=minority_t,
        exclude_classes=EXCLUDE_CLASSES,
    )
    val_dataset = CornDataset(
        csv_path=str(SPLITS_DIR / "val.csv"),
        transform=val_t,
        exclude_classes=EXCLUDE_CLASSES,
    )
    test_dataset = CornDataset(
        csv_path=str(SPLITS_DIR / "test.csv"),
        transform=test_t,
        exclude_classes=EXCLUDE_CLASSES,
    )

    sampler = build_weighted_sampler(train_dataset)
    class_weights = build_class_weights(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=sampler,       # incompatible con shuffle=True
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # TODO: definir arquitectura, optimizador y loop de entrenamiento


if __name__ == "__main__":
    main()
