import logging
import os

import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset, WeightedRandomSampler

from src.config import DATASET_ROOT, PROJECT_ROOT
from src.data.loader import load_and_normalize_image
from src.data.transforms import MINORITY_CLASSES

_DEFAULT_CONFIG = str(PROJECT_ROOT / "config" / "dataset.yaml")

logger = logging.getLogger(__name__)


class CornDataset(Dataset):
    """
    Componente Dataset personalizado para el mapeo y consumo indexado
    de imágenes de patologías y deficiencias en hojas de maíz.
    """

    def __init__(
        self,
        csv_path: str,
        config_path: str = _DEFAULT_CONFIG,
        transform=None,
        minority_transform=None,
        exclude_classes: list[str] | None = None,
    ):
        """
        Args:
            csv_path: Ruta al manifiesto del split (train.csv, val.csv o test.csv).
            config_path: Ruta al archivo de configuración paramétrica.
            transform: Pipeline de transformaciones estándar (torchvision).
            minority_transform: Pipeline extendido aplicado a clases en MINORITY_CLASSES.
                                Si None, todas las muestras usan `transform`.
            exclude_classes: Clases a excluir del dataset en tiempo de construcción.
                             El CSV permanece inmutable; la exclusión es una decisión de pipeline.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No se encontró el archivo de manifiesto: {csv_path}")

        self.transform = transform
        self.minority_transform = minority_transform
        self.dataset_root = DATASET_ROOT

        # 1. Cargar y filtrar el manifiesto
        df = pd.read_csv(csv_path)
        if exclude_classes:
            df = df[~df["label"].isin(exclude_classes)].reset_index(drop=True)
        self.data_frame = df

        # 2. Cargar mapeo de clases desde la configuración centralizada
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Construir class_to_idx compacto solo con las clases presentes tras el filtro.
        # El orden respeta la lista del YAML para reproducibilidad entre ejecuciones.
        present = set(self.data_frame["label"].unique())
        self.allowed_classes = [c for c in config["dataset"]["classes"] if c in present]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.allowed_classes)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        # Validar que no haya etiquetas en el CSV no cubiertas por el YAML.
        # Falla en construcción, no en el primer batch.
        unknown = present - set(c for c in config["dataset"]["classes"])
        if unknown:
            raise ValueError(f"Etiquetas en el CSV no registradas en config: {sorted(unknown)}")

    def __len__(self) -> int:
        """Devuelve el tamaño neto total de la muestra actual."""
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Obtiene y procesa un elemento del dataset en caliente bajo demanda.

        Garantiza:
        1. Carga perezosa desde disco duro para optimizar la memoria RAM.
        2. Normalización física y corrección de color/rotación en el milisegundo de lectura.
        3. Transformación dimensional homogénea y conversión a tensor para la GPU.
        """
        # Extraer metadatos de la fila correspondiente del CSV
        img_path = self.dataset_root / self.data_frame.iloc[idx]["image_path"]
        class_name = self.data_frame.iloc[idx]["label"]

        # 1. Carga y normalización en caliente del formato (RGB, corrección EXIF de smartphones)
        try:
            image = load_and_normalize_image(img_path)
        except (FileNotFoundError, RuntimeError) as e:
            # Si una imagen del manifiesto está corrupta o desaparece en tiempo de entrenamiento,
            # devolvemos el siguiente índice válido en lugar de matar el DataLoader worker.
            logger.warning(
                f"Imagen no disponible en idx={idx} ({img_path}): {e}. Usando idx={idx + 1}."
            )
            return self[idx + 1 if idx + 1 < len(self) else 0]

        # 2. Mapear la etiqueta de texto a su correspondiente índice entero codificado
        label_idx = self.class_to_idx[class_name]

        # 3. Seleccionar pipeline: extendido para clases minoritarias, estándar para el resto
        pipeline = (
            self.minority_transform
            if self.minority_transform is not None and class_name in MINORITY_CLASSES
            else self.transform
        )
        if pipeline:
            image = pipeline(image)

        return image, label_idx


def build_weighted_sampler(dataset: "CornDataset", seed: int) -> WeightedRandomSampler:
    labels = dataset.data_frame["label"].tolist()
    class_counts = dataset.data_frame["label"].value_counts().to_dict()
    sample_weights = torch.tensor(
        [1.0 / class_counts[label] for label in labels], dtype=torch.float
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
