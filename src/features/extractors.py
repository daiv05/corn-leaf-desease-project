import logging
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

from src.config import PROJECT_ROOT

_DEFAULT_CONFIG = str(PROJECT_ROOT / "config" / "dataset.yaml")

logger = logging.getLogger(__name__)


class FeatureExtractor(ABC):
    """Interfaz abstracta para extractores de características visuales."""

    @abstractmethod
    def extract(self, image: Image.Image) -> np.ndarray:
        """
        Extrae un vector de características de la imagen.

        Args:
            image: PIL.Image en modo RGB, ya redimensionada al tamaño objetivo.

        Returns:
            np.ndarray 1D de dtype float32.
        """

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimensión del vector de salida."""


class HOGExtractor(FeatureExtractor):
    """Histograma de Gradientes Orientados sobre imagen en escala de grises."""

    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: tuple[int, int] = (8, 8),
        cells_per_block: tuple[int, int] = (2, 2),
    ) -> None:
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self._dim: int | None = None

    def extract(self, image: Image.Image) -> np.ndarray:
        gray = np.array(image.convert("L"))
        features = hog(
            gray,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            feature_vector=True,
        )
        self._dim = features.size
        return features.astype(np.float32)

    @property
    def feature_dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("Llama a extract() al menos una vez para conocer feature_dim.")
        return self._dim


class HSVHistogramExtractor(FeatureExtractor):
    """Histograma de color en espacio HSV concatenado por canal."""

    def __init__(self, bins: list[int] | None = None) -> None:
        self.bins = bins if bins is not None else [16, 16, 16]

    def extract(self, image: Image.Image) -> np.ndarray:
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        ranges = [(0, 180), (0, 256), (0, 256)]
        histograms = []
        for ch, (n_bins, (lo, hi)) in enumerate(zip(self.bins, ranges)):
            hist = cv2.calcHist([hsv], [ch], None, [n_bins], [lo, hi])
            cv2.normalize(hist, hist)
            histograms.append(hist.flatten())
        return np.concatenate(histograms).astype(np.float32)

    @property
    def feature_dim(self) -> int:
        return sum(self.bins)


class LBPExtractor(FeatureExtractor):
    """Patrón Binario Local uniforme — descriptor de textura rotacionalmente invariante."""

    def __init__(self, radius: int = 3, n_points: int = 24) -> None:
        self.radius = radius
        self.n_points = n_points
        # Uniform LBP produce n_points + 2 patrones distintos
        self._dim = n_points + 2

    def extract(self, image: Image.Image) -> np.ndarray:
        gray = np.array(image.convert("L"))
        lbp = local_binary_pattern(gray, self.n_points, self.radius, method="uniform")
        hist, _ = np.histogram(
            lbp.ravel(), bins=self._dim, range=(0, self._dim), density=True
        )
        return hist.astype(np.float32)

    @property
    def feature_dim(self) -> int:
        return self._dim


class GLCMExtractor(FeatureExtractor):
    """
    Matriz de Co-ocurrencia de Niveles de Gris.

    Extrae contraste, correlación, energía y homogeneidad, promediados
    sobre todas las combinaciones de distancias y ángulos.
    """

    def __init__(
        self,
        distances: list[int] | None = None,
        angles: list[float] | None = None,
        properties: list[str] | None = None,
    ) -> None:
        self.distances = distances if distances is not None else [1, 3]
        self.angles = angles if angles is not None else [0, 0.785, 1.571, 2.356]
        self.properties = properties if properties is not None else [
            "contrast", "correlation", "energy", "homogeneity"
        ]

    def extract(self, image: Image.Image) -> np.ndarray:
        gray = np.array(image.convert("L"), dtype=np.uint8)
        glcm = graycomatrix(
            gray,
            distances=self.distances,
            angles=self.angles,
            levels=256,
            symmetric=True,
            normed=True,
        )
        features = [
            graycoprops(glcm, prop).mean()
            for prop in self.properties
        ]
        return np.array(features, dtype=np.float32)

    @property
    def feature_dim(self) -> int:
        return len(self.properties)


class CombinedExtractor(FeatureExtractor):
    """
    Concatena HOG, HSV, LBP y GLCM en un único vector de características.

    Uso típico: CombinedExtractor.from_config('config/dataset.yaml')
    """

    def __init__(
        self,
        hog: HOGExtractor,
        hsv: HSVHistogramExtractor,
        lbp: LBPExtractor,
        glcm: GLCMExtractor,
        target_size: tuple[int, int] = (224, 224),
    ) -> None:
        self._hog = hog
        self._hsv = hsv
        self._lbp = lbp
        self._glcm = glcm
        self.target_size = target_size

    @classmethod
    def from_config(cls, config_path: str = _DEFAULT_CONFIG) -> "CombinedExtractor":
        """Instancia todos los extractores con parámetros declarados en el YAML."""
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        feat = cfg["features"]
        w, h = feat["target_size"]

        return cls(
            hog=HOGExtractor(
                orientations=feat["hog"]["orientations"],
                pixels_per_cell=tuple(feat["hog"]["pixels_per_cell"]),
                cells_per_block=tuple(feat["hog"]["cells_per_block"]),
            ),
            hsv=HSVHistogramExtractor(bins=feat["hsv"]["bins"]),
            lbp=LBPExtractor(
                radius=feat["lbp"]["radius"],
                n_points=feat["lbp"]["n_points"],
            ),
            glcm=GLCMExtractor(
                distances=feat["glcm"]["distances"],
                angles=feat["glcm"]["angles"],
                properties=feat["glcm"]["properties"],
            ),
            target_size=(w, h),
        )

    def extract(self, image: Image.Image) -> np.ndarray:
        """
        Redimensiona la imagen y extrae el vector combinado de características.

        Garantías:
        1. La imagen se redimensiona a target_size antes de cualquier extractor.
        2. Los cuatro descriptores se concatenan en el mismo orden siempre.
        3. El vector resultante es float32 y 1D.
        """
        image = image.resize(self.target_size, Image.LANCZOS)
        parts = [
            self._hog.extract(image),
            self._hsv.extract(image),
            self._lbp.extract(image),
            self._glcm.extract(image),
        ]
        return np.concatenate(parts).astype(np.float32)

    @property
    def feature_dim(self) -> int:
        return (
            self._hsv.feature_dim
            + self._lbp.feature_dim
            + self._glcm.feature_dim
            # HOG requiere una extracción previa para conocer su dimensión
        )
