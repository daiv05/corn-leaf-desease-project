"""Validación dejando una fuente fuera en GPU de Modal.

Orquesta por subprocess el mismo CLI que corre en local
(scripts/experiments/leave_one_source_out.py), heredando el entorno de la imagen
(DATASET_ROOT=/data, OUTPUT_ROOT=/outputs) para que los Volumes montados resuelvan igual
que en el pipeline principal.

Uso:
    modal run scripts/modal/leave_one_source_out.py
    modal run scripts/modal/leave_one_source_out.py --folds maize-diseases
    modal run scripts/modal/leave_one_source_out.py --arm border_ring
"""

import subprocess
import sys
from pathlib import Path

import modal

from scripts.modal._common import REPO_ANCHOR, dataset_vol, image, outputs_vol

app = modal.App("corn-leave-one-source-out", image=image)

RESULTS_TEMPLATE = "/outputs/experiments/leave_one_source_out{tag}{arm}{seed}.json"


@app.function(
    gpu="A10",
    # El deduplicado inicial hashea las 33k imagenes una sola vez y el DataLoader decodifica
    # JPEG en cada epoca: ambos son I/O-bound y 8 cores alimentan holgadamente a la A10.
    cpu=8.0,
    volumes={"/data": dataset_vol, "/outputs": outputs_vol},
    secrets=[modal.Secret.from_name("hf")],
    # Once pliegues, cada uno hasta 25 epocas sobre ~9k imagenes de train capadas, mas el
    # hashing perceptual inicial. El techo cubre el peor caso sin early stopping.
    timeout=8 * 3600,
)
def run_leave_one_source_out(
    model: str = "efficientnet_lite0",
    epochs: int = 25,
    train_cap: int = 1000,
    val_cap: int = 2000,
    batch_size: int = 64,
    seed: int = 0,
    folds: str = "",
    arm: str = "original",
    balance_groups: bool = False,
    backmix: float = 0.0,
) -> None:
    """Ejecuta la validación por fuente en la GPU remota y persiste el JSON en el Volume.

    @param {int} train_cap Máximo de imágenes de entrenamiento por clase.
    @param {int} val_cap Máximo de imágenes de la fuente de validación.
    @param {str} folds Lista separada por comas de fuentes a evaluar; vacío ejecuta todas.
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
        "scripts/experiments/leave_one_source_out.py",
        "--splits-dir", str(splits_dir),
        "--model", model,
        "--epochs", str(epochs),
        "--train-cap", str(train_cap),
        "--val-cap", str(val_cap),
        "--batch-size", str(batch_size),
        "--seed", str(seed),
        "--arm", arm,
        *(["--balance-groups"] if balance_groups else []),
        *(["--backmix", str(backmix)] if backmix else []),
        "--num-workers", "8",
        "--output", RESULTS_TEMPLATE.format(
            tag=("_balanced" if balance_groups else "")
            + (f"_backmix{backmix:g}" if backmix else ""),
            arm="" if arm == "original" else f"_{arm}",
            seed="" if seed == 0 else f"_seed{seed}"),
    ]
    if folds:
        command += ["--folds", folds]

    subprocess.run(command, check=True, cwd=REPO_ANCHOR)
    outputs_vol.commit()


@app.local_entrypoint()
def main(
    model: str = "efficientnet_lite0",
    epochs: int = 25,
    train_cap: int = 1000,
    val_cap: int = 2000,
    batch_size: int = 64,
    seed: int = 0,
    folds: str = "",
    arm: str = "original",
    balance_groups: bool = False,
    backmix: float = 0.0,
) -> None:
    """Entrypoint de `modal run`: dispara la validación por fuente en la GPU remota."""
    run_leave_one_source_out.remote(
        model=model,
        epochs=epochs,
        train_cap=train_cap,
        val_cap=val_cap,
        batch_size=batch_size,
        seed=seed,
        folds=folds,
        arm=arm,
        balance_groups=balance_groups,
        backmix=backmix,
    )
