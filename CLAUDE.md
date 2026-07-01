# CLAUDE.md - Corn Leaf Disease Project

## Reglas de datos

- **Nunca modificar `raw/`.** Es inmutable, solo fuente original.
- `clean/` es la única fuente de verdad para entrenamiento. Estructura: `clean/<clase>/{lab,real}/`.
- Los CSV de `splits/` son derivados reproducibles (`make splits` / `make splits-baseline`). No editarlos a mano.

## Arquitectura (`src/`)

- **Único punto de entrada a imagen:** `load_and_normalize_image()` (`src/data/loader.py`) — corrección EXIF + RGB antes de cualquier transform.
- **Config centralizada:** `config/dataset.yaml` declara clases, `target_size`, seed y el perfil `baseline`. Los módulos lo leen; nunca hardcodear constantes de dominio.
- **Sin `sys.path.append`.** Paquete editable (`pip install -e .`); los imports `src.*` resuelven directo.
- **`MINORITY_CLASSES`** (`src/data/transforms.py`) es un `frozenset` estático que `CornDataset.__getitem__` consulta para aplicar augmentation extendido — independiente del subset/límite usado al generar los splits.
- Para ubicar símbolos, llamadas o impacto de cambios en `src/`, usa CodeGraph (si está disponible) en vez de grep/lectura manual.

## Pipelines

- **Datos:** `clean/<clase>/{lab,real}/` → `create_splits.py` (valida integridad PIL, deduplica por SHA-256, estratifica por `label+environment`) → `splits/seed_42/` (9 clases) o `splits/seed_42_baseline/` (`--baseline`, subset de `config/dataset.yaml -> baseline:`).
- **Baselines (funcional, PyTorch):** `CornDataset` → `WeightedRandomSampler` → `DataLoader` → `MODEL_REGISTRY.build(<efficientnet_b0|efficientnet_lite0|mobilenet_v3_large>)` vía `train_baselines.py`. Pese al nombre, no es un pipeline sklearn — es DL completo, pensado para comparar arquitecturas rápido y barato.
- **Principal (`train.py`):** comparte toda la infraestructura de datos/modelos con baselines; el loop de entrenamiento está pendiente de implementar.

## Clases del dataset

Definidas en `config/dataset.yaml -> dataset.classes` (orden canónico para `class_to_idx`). Minoritarias:
`common_rust` (3.9x), `gray_leaf_spot` (7.9x), `nitrogen_deficiency` (16.8x), `phosphorus_deficiency` (14.3x), `potassium_deficiency` (32.9x).
El perfil `baseline` usa por defecto `healthy`, `common_rust`, `fall_armyworm`, `nitrogen_deficiency` (500 img/clase).

## Dataset: hosting y descarga

`clean/` (~25k imágenes) vive en Hugging Face Datasets Hub (fuente primaria) con Google Drive de respaldo;
`download_dataset.py --source auto` resuelve cuál usar. `scripts/download_datasets.sh` es un flujo distinto:
ingesta de fuentes crudas nuevas (Kaggle/Mendeley/Roboflow) hacia `raw/`, no toca `clean/`.

## Entrenamiento en GPU remota (vast.ai)

Ver guía completa en `docs/es/deployment/vast-ai.md`. Resumen: `Dockerfile` (Python 3.11 + PyTorch CUDA,
instalado en `venv/`) + `scripts/vastai/onstart.sh` (provisioning: clona, instala, descarga dataset) +
`scripts/vastai/launch.py` (wrapper sobre la CLI `vastai`: search/create/run/sync/destroy).

## Comandos frecuentes

```bash
make install                          # pip install -e ".[dev,analysis]"
make download-dataset                 # clean/ (HF Hub, fallback Google Drive)
make splits / make splits-baseline    # regenera splits CSV
make train-baselines [MODELS=<nombre>] / make train-baselines-full
make train                            # loop de entrenamiento pendiente
make summary                          # conteo de imágenes por clase/entorno
make lint / make fmt                  # ruff check / ruff format
```

## Setup local

Ver [LOCAL.md](LOCAL.md) para levantar el proyecto (venv, `.env`, descarga del dataset).
