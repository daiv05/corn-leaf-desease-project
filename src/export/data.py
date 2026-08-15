"""Construcción del loader de test compartido por exportación y evaluación.

Ambos entrypoints necesitan exactamente el mismo split, el mismo `class_to_idx` y el mismo
pipeline de transforms ('test', determinista y sin augmentation). Factorizarlo evita que
la paridad y la evaluación terminen mirando datos distintos.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from src.config import get_output_root
from src.data.dataset import CornDataset
from src.data.transforms import CornTransformFactory

logger = logging.getLogger(__name__)


def resolve_test_csv(run_dir: Path, splits_dir: str | None) -> Path:
    """
    Resuelve la ruta de test.csv: --splits-dir si se pasó, si no el de summary.json.

    @param {Path} run_dir Directorio del run.
    @param {str|None} splits_dir Override explícito del directorio de splits.
    @returns {Path} Ruta a test.csv.
    @throws {SystemExit} Si el archivo no existe.
    """
    if splits_dir:
        resolved = Path(splits_dir)
    else:
        summary = json.loads((run_dir / "summary.json").read_text())
        resolved = Path(
            summary.get("splits_dir", get_output_root() / "splits" / "seed_42")
        )
    test_csv = resolved / "test.csv"
    if not test_csv.exists():
        raise SystemExit(
            f"No existe {test_csv}. Pasa --splits-dir con el directorio correcto."
        )
    return test_csv


def build_test_loader(
    test_csv: Path,
    config_path: Path,
    class_to_idx: dict[str, int],
    image_size: tuple[int, int],
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, pd.Series]:
    """
    Construye el DataLoader del split de test y la serie de entornos alineada.

    `shuffle=False` no es opcional: la serie de entornos se alinea por posición con el
    orden en que el loader entrega las imágenes.

    @param {Path} test_csv Ruta al CSV del split de test.
    @param {Path} config_path Ruta al YAML de configuración.
    @param {dict[str,int]} class_to_idx Mapeo clase->índice del run.
    @param {tuple[int,int]} image_size Alto y ancho de entrada (h, w).
    @param {int} batch_size Tamaño de batch.
    @param {int} num_workers Workers del DataLoader.
    @returns {tuple[DataLoader, pd.Series]} Loader y entornos por imagen.
    """
    factory = CornTransformFactory(config_path=str(config_path), target_size=image_size)
    dataset = CornDataset(
        csv_path=str(test_csv),
        config_path=str(config_path),
        transform=factory.get_pipeline("test"),
        class_to_idx=class_to_idx,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    # Se lee del DataFrame del propio dataset, no del CSV: si algún día CornDataset
    # filtra filas, la serie sigue alineada con lo que entrega el loader.
    frame = dataset.data_frame
    environments = (
        frame["environment"].reset_index(drop=True)
        if "environment" in frame.columns
        else pd.Series(dtype=str)
    )
    return loader, environments
