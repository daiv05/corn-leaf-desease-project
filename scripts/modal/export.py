"""Exportacion de checkpoints del pipeline principal a ONNX/TFLite, en Modal.

Espeja scripts/pipeline/export.py: orquesta por subprocess el mismo CLI, leyendo
checkpoints/splits ya persistidos en el Volume corn-outputs por train_main.

Uso:
    modal run scripts/modal/export.py::export_main --models "efficientnet_b0"
    modal run scripts/modal/export.py::export_main --models "shufflenet_v2_x1_0" \
        --formats "onnx,tflite"
    modal run scripts/modal/export.py::export_main --formats "tflite" --quantize "int8"
    modal run scripts/modal/export.py::evaluate_export_main --formats "onnx,tflite"
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
    quantize: str = "",
    splits_dir: str = "",
    tolerance: float = 0.0,
    min_agreement_rate: float = 0.0,
    parity_sample_size: int = 0,
    no_parity: bool = False,
) -> None:
    """
    Exporta checkpoints del pipeline principal a ONNX/TFLite. Espeja `make export-main`.

    @param {str} models Modelos separados por espacio.
    @param {str} run run_id especifico; vacio usa latest.json.
    @param {str} checkpoint Ruta explicita a un checkpoint .pth (ignora run si se pasa).
    @param {str} formats Formatos a exportar, CSV (ej: "onnx,tflite").
    @param {str} quantize Cuantizacion: "int8" o vacio para FP32.
    @param {str} splits_dir Directorio con test.csv; vacio usa el de summary.json.
    @param {float} tolerance Tolerancia de paridad; 0.0 usa el default del script.
    @param {float} min_agreement_rate Acuerdo minimo; 0.0 usa el default del script.
    @param {int} parity_sample_size Muestras para paridad; 0 usa el default del script.
    @param {bool} no_parity Omite la validacion de paridad numerica.
    """
    dataset_vol.reload()
    args = [
        sys.executable,
        "scripts/pipeline/export.py",
        "--models",
        *models.split(),
        "--output-dir",
        f"{OUTPUTS_MOUNT}/main",
        "--formats",
        formats,
    ]
    if quantize:
        args += ["--quantize", quantize]
    if run:
        args += ["--run", run]
    if checkpoint:
        args += ["--checkpoint", checkpoint]
    if splits_dir:
        args += ["--splits-dir", splits_dir]
    if tolerance:
        args += ["--tolerance", str(tolerance)]
    if min_agreement_rate:
        args += ["--min-agreement-rate", str(min_agreement_rate)]
    if parity_sample_size:
        args += ["--parity-sample-size", str(parity_sample_size)]
    if no_parity:
        args.append("--no-parity")
    try:
        subprocess.run(args, check=True, cwd=REPO_ANCHOR)
    finally:
        # Se commitea aunque un modelo falle: los que si exportaron deben persistir.
        outputs_vol.commit()


@app.function(
    volumes={"/data": dataset_vol, "/outputs": outputs_vol},
    secrets=[modal.Secret.from_name("hf")],
    timeout=7200,
)
def evaluate_export_main(
    models: str = DEFAULT_MODELS,
    run: str = "",
    formats: str = "onnx",
    quantize: str = "",
    splits_dir: str = "",
    batch_size: int = 0,
    no_torch_baseline: bool = False,
    max_macro_f1_drop: float = 0.0,
) -> None:
    """
    Evalua modelos exportados sobre el split de test completo. Espeja `make eval-export-main`.

    @param {str} models Modelos separados por espacio.
    @param {str} run run_id especifico; vacio usa latest.json.
    @param {str} formats Formatos a evaluar, CSV (ej: "onnx,tflite").
    @param {str} quantize Variante a evaluar: "int8" o vacio para FP32.
    @param {str} splits_dir Directorio con test.csv; vacio usa el de summary.json.
    @param {int} batch_size Tamano de batch; 0 usa el default del script.
    @param {bool} no_torch_baseline Omite la referencia PyTorch (sin deltas).
    @param {float} max_macro_f1_drop Caida maxima de macro-F1; 0.0 usa el default.
    """
    dataset_vol.reload()
    outputs_vol.reload()
    args = [
        sys.executable,
        "scripts/pipeline/evaluate_export.py",
        "--models",
        *models.split(),
        "--output-dir",
        f"{OUTPUTS_MOUNT}/main",
        "--formats",
        formats,
    ]
    if quantize:
        args += ["--quantize", quantize]
    if run:
        args += ["--run", run]
    if splits_dir:
        args += ["--splits-dir", splits_dir]
    if batch_size:
        args += ["--batch-size", str(batch_size)]
    if no_torch_baseline:
        args.append("--no-torch-baseline")
    if max_macro_f1_drop:
        args += ["--max-macro-f1-drop", str(max_macro_f1_drop)]
    try:
        subprocess.run(args, check=True, cwd=REPO_ANCHOR)
    finally:
        outputs_vol.commit()
