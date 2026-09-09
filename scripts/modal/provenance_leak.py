"""Test de fuga de procedencia en GPU de Modal.

Orquesta por subprocess el mismo CLI que corre en local
(scripts/experiments/provenance_leak.py), heredando el entorno de la imagen
(DATASET_ROOT=/data, OUTPUT_ROOT=/outputs) para que los Volumes montados resuelvan igual
que en el pipeline principal. Consume splits/seed_42 ya presente en el Volume corn-outputs.

Uso:
    modal run scripts/modal/provenance_leak.py
    modal run scripts/modal/provenance_leak.py --arms original,border_ring --seeds 0,1,2
    modal run scripts/modal/provenance_leak.py::fetch_results
"""

import subprocess
import sys
from pathlib import Path

import modal

from scripts.modal._common import REPO_ANCHOR, dataset_vol, image, outputs_vol

app = modal.App("corn-provenance-leak", image=image)

RESULTS_PATH = "/outputs/experiments/provenance_leak.json"


@app.function(
    gpu="A10",
    # Los brazos son transformaciones baratas sobre la imagen decodificada: el cuello es la
    # decodificacion JPEG del DataLoader, no la GPU. 8 cores alimentan holgadamente a la A10.
    cpu=8.0,
    volumes={"/data": dataset_vol, "/outputs": outputs_vol},
    secrets=[modal.Secret.from_name("hf")],
    # Tres brazos x tres semillas x hasta 25 epocas sobre ~7.7k imagenes de train capadas.
    # Medido en local a ~85 s/epoca sin cache; en A10 con 8 workers baja bastante, pero el
    # techo cubre el peor caso sin early stopping.
    timeout=6 * 3600,
)
def run_leak_probe(
    arms: str = "original,border_ring,center_only",
    seeds: str = "0,1,2",
    model: str = "efficientnet_lite0",
    epochs: int = 25,
    train_cap: int = 1000,
    ring_fraction: float = 0.10,
    batch_size: int = 64,
) -> None:
    """Ejecuta el experimento en la GPU remota y persiste el JSON en el Volume.

    @param {str} arms Brazos separados por coma.
    @param {str} seeds Semillas separadas por coma.
    @param {int} train_cap Máximo de imágenes de entrenamiento por clase.
    @param {float} ring_fraction Grosor del anillo de borde como fracción del lado.
    """
    dataset_vol.reload()
    outputs_vol.reload()

    splits_dir = Path("/outputs/splits/seed_42")
    if not (splits_dir / "test.csv").exists():
        raise FileNotFoundError(
            f"No existe {splits_dir}/test.csv en el Volume corn-outputs; "
            "corre antes `modal run scripts/modal/train.py::make_splits`"
        )

    command = [
        sys.executable,
        "scripts/experiments/provenance_leak.py",
        "--splits-dir", str(splits_dir),
        "--arms", arms,
        "--seeds", seeds,
        "--model", model,
        "--epochs", str(epochs),
        "--train-cap", str(train_cap),
        "--ring-fraction", str(ring_fraction),
        "--batch-size", str(batch_size),
        "--num-workers", "8",
        "--output", RESULTS_PATH,
    ]
    subprocess.run(command, check=True, cwd=REPO_ANCHOR)
    outputs_vol.commit()


@app.function(volumes={"/outputs": outputs_vol}, timeout=600)
def fetch_results() -> str:
    """Devuelve el JSON de resultados para volcarlo en local."""
    outputs_vol.reload()
    path = Path(RESULTS_PATH)
    if not path.exists():
        raise FileNotFoundError(f"No hay resultados en {RESULTS_PATH}")
    return path.read_text(encoding="utf-8")


@app.local_entrypoint()
def main(
    arms: str = "original,border_ring,center_only",
    seeds: str = "0,1,2",
    model: str = "efficientnet_lite0",
    epochs: int = 25,
    train_cap: int = 1000,
    ring_fraction: float = 0.10,
    batch_size: int = 64,
) -> None:
    """Entrypoint de `modal run`: dispara el test de fuga en la GPU remota."""
    run_leak_probe.remote(
        arms=arms,
        seeds=seeds,
        model=model,
        epochs=epochs,
        train_cap=train_cap,
        ring_fraction=ring_fraction,
        batch_size=batch_size,
    )
