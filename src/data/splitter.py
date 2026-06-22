import logging
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DatasetSplitter(ABC):
    """Interfaz abstracta para la partición de conjuntos de datos (DIP)."""

    @abstractmethod
    def split(
        self, data_manifest: pd.DataFrame, train_size: float, val_size: float, test_size: float
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pass


class HierarchicalStratifiedSplitter(DatasetSplitter):
    """Ejecuta una división estratificada considerando combinaciones de Clase y Entorno."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def split(
        self, data_manifest: pd.DataFrame, train_size: float, val_size: float, test_size: float
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if not abs((train_size + val_size + test_size) - 1.0) < 1e-9:
            raise ValueError("Las proporciones de train, val y test deben sumar exactamente 1.0")

        # Generar súper-etiqueta temporal que fusiona la patología y el entorno de captura
        # Ejemplo: 'common_rust_real' o 'common_rust_lab'
        stratify_col = data_manifest["label"] + "_" + data_manifest["environment"]

        # Ajustar el tamaño proporcional del segundo split
        relative_test_size = test_size / (val_size + test_size)

        # Primer Split: Separar Entrenamiento del bloque de evaluación remanente
        train_df, temp_df = train_test_split(
            data_manifest, train_size=train_size, stratify=stratify_col, random_state=self.seed
        )

        # Recalcular la súper-etiqueta en el bloque remanente
        temp_stratify = temp_df["label"] + "_" + temp_df["environment"]

        # Segundo Split: Particionar de forma homogénea Validación y Prueba estrictos
        val_df, test_df = train_test_split(
            temp_df, test_size=relative_test_size, stratify=temp_stratify, random_state=self.seed
        )

        return train_df, val_df, test_df
