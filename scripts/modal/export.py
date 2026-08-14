"""Exportacion de checkpoints del pipeline principal a ONNX/TFLite, en Modal.

Espeja scripts/pipeline/export.py: orquesta por subprocess el mismo CLI, leyendo
checkpoints/splits ya persistidos en el Volume corn-outputs por train_main.

Uso:
    modal run scripts/modal/export.py::export_main --models "efficientnet_b0"
    modal run scripts/modal/export.py::export_main --models "shufflenet_v2_x1_0" \
        --formats "onnx,tflite"
Requiere: `pip install -e ".[cloud]"`, `modal setup`, y el secret:
    modal secret create hf HF_TOKEN=hf_xxx
"""

import subprocess
import sys

import modal

from scripts.modal._common import (
    DEFAULT_MODELS,
    OUTPUTS_MOUNT,
    REPO_ANCHOR,
    dataset_vol,
    image,
    outputs_vol,
)

app = modal.App("corn-leaf-export", image=image)


@app.function(
    volumes={"/data": dataset_vol, "/outputs": outputs_vol},
    secrets=[modal.Secret.from_name("hf")],
    timeout=3600,
)
def export_main(
    models: str = DEFAULT_MODELS,
    run: str = "",
    checkpoint: str = "",
    formats: str = "onnx",
    splits_dir: str = "",
    tolerance: float = 0.0,
    parity_sample_size: int = 0,
    no_parity: bool = False,
) -> None:
    """
    Exporta checkpoints del pipeline principal a ONNX/TFLite. Espeja `make export-main`.

    Itera `models.split()` porque export.py opera sobre un modelo a la vez
    (--model singular), a diferencia de train_main.

    @param {str} models Modelos separados por espacio.
    @param {str} run run_id especifico; vacio usa latest.json.
    @param {str} checkpoint Ruta explicita a un checkpoint .pth (ignora run si se pasa).
    @param {str} formats Formatos a exportar, CSV (ej: "onnx,tflite").
    @param {str} splits_dir Directorio con test.csv; vacio usa el de summary.json.
    @param {float} tolerance Tolerancia de paridad; 0.0 usa el default del script.
    @param {int} parity_sample_size Muestras para paridad; 0 usa el default del script.
    @param {bool} no_parity Omite la validacion de paridad numerica.
    """
    dataset_vol.reload()
    for model_name in models.split():
        args = [
            sys.executable,
            "scripts/pipeline/export.py",
            "--model",
            model_name,
            "--output-dir",
            f"{OUTPUTS_MOUNT}/main",
            "--formats",
            formats,
        ]
        if run:
            args += ["--run", run]
        if checkpoint:
            args += ["--checkpoint", checkpoint]
        if splits_dir:
            args += ["--splits-dir", splits_dir]
        if tolerance:
            args += ["--tolerance", str(tolerance)]
        if parity_sample_size:
            args += ["--parity-sample-size", str(parity_sample_size)]
        if no_parity:
            args.append("--no-parity")
        subprocess.run(args, check=True, cwd=REPO_ANCHOR)
    outputs_vol.commit()
