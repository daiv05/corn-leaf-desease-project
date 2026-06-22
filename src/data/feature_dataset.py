import logging
import os

import numpy as np
import pandas as pd
import yaml

from src.config import DATASET_ROOT, PROJECT_ROOT
from src.data.loader import load_and_normalize_image
from src.features.extractors import FeatureExtractor

_DEFAULT_CONFIG = str(PROJECT_ROOT / "config" / "dataset.yaml")

logger = logging.getLogger(__name__)


class CornFeatureDataset:
    """
    Dataset de características para el pipeline de baselines (sklearn).

    A diferencia de CornDataset, devuelve (np.ndarray, int) en lugar de
    (torch.Tensor, int), lo que permite alimentar directamente a modelos sklearn
    sin pasar por DataLoader.
    """

    def __init__(
        self,
        csv_path: str,
        extractor: FeatureExtractor,
        config_path: str = _DEFAULT_CONFIG,
        exclude_classes: list[str] | None = None,
    ) -> None:
        """
        Args:
            csv_path: Ruta al manifiesto del split (train.csv, val.csv o test.csv).
            extractor: Instancia de FeatureExtractor que transforma PIL.Image -> np.ndarray.
            config_path: Ruta al archivo de configuración paramétrica.
            exclude_classes: Clases a excluir del dataset en tiempo de construcción.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No se encontró el archivo de manifiesto: {csv_path}")

        self.extractor = extractor
        self.dataset_root = DATASET_ROOT

        df = pd.read_csv(csv_path)
        if exclude_classes:
            df = df[~df["label"].isin(exclude_classes)].reset_index(drop=True)
        self.data_frame = df

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        present = set(self.data_frame["label"].unique())
        self.allowed_classes = [c for c in config["dataset"]["classes"] if c in present]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.allowed_classes)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        unknown = present - set(config["dataset"]["classes"])
        if unknown:
            raise ValueError(f"Etiquetas en el CSV no registradas en config: {sorted(unknown)}")

    def __len__(self) -> int:
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        """
        Carga una imagen y devuelve su vector de características junto con su etiqueta.

        Garantías:
        1. La imagen se carga con corrección EXIF y conversión RGB.
        2. El extractor recibe siempre un PIL.Image en modo RGB.
        3. Si la imagen no está disponible, se usa el siguiente índice válido.
        """
        img_path = self.dataset_root / self.data_frame.iloc[idx]["image_path"]
        class_name = self.data_frame.iloc[idx]["label"]

        try:
            image = load_and_normalize_image(img_path)
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning(
                f"Imagen no disponible en idx={idx} ({img_path}): {e}. "
                f"Usando idx={idx + 1}."
            )
            return self[idx + 1 if idx + 1 < len(self) else 0]

        features = self.extractor.extract(image)
        label_idx = self.class_to_idx[class_name]
        return features, label_idx

    def load_all(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Carga todo el split en memoria y devuelve (X, y) listos para sklearn.

        Returns:
            X: np.ndarray de shape (n_samples, n_features), dtype float32.
            y: np.ndarray de shape (n_samples,), dtype int64.
        """
        X_list: list[np.ndarray] = []
        y_list: list[int] = []

        for idx in range(len(self)):
            features, label = self[idx]
            X_list.append(features)
            y_list.append(label)

        X = np.vstack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.int64)
        return X, y
