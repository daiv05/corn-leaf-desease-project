# CLAUDE.md - Corn Leaf Disease Project

## Reglas de datos

- **Nunca modificar `raw/`.** La carpeta `$DATASET_ROOT/raw` es inmutable y solo sirve como fuente original.
- `clean/` es la única fuente de verdad para el pipeline de entrenamiento. Estructura fija: `clean/<clase>/{lab,real}/`.
- Los splits CSV (`splits/seed_42/`, `splits/seed_42_baseline/`) son derivados reproducibles: se regeneran con `make splits` / `make splits-baseline`. No editarlos a mano.

---

## Estructura del proyecto

```
corn-leaf-desease-project/
├── config/
│   └── dataset.yaml          # Parámetros centralizados: clases, tamaños, seed, perfil baseline
├── docs/                     # Documentación VitePress (es/), incluye docs/es/deployment/vast-ai.md
├── fiftyone/                 # Scripts de exploración visual con FiftyOne
├── notebooks/                # EDA en Jupyter y otras notebooks de análisis
├── scripts/
│   ├── cleanup/               # Scripts de limpieza por clase/dataset (one-shot, ya ejecutados)
│   ├── dataset/
│   │   ├── upload_to_hf.py    # Sube clean/ a un repo de tipo dataset en Hugging Face Hub
│   │   └── download_dataset.py # Descarga clean/ (HF Hub primero, Google Drive de fallback)
│   ├── download_datasets.sh   # Ingesta de fuentes crudas nuevas (Kaggle/Mendeley/Roboflow) a raw/
│   ├── pipeline/
│   │   ├── create_splits.py   # Entrypoint: genera manifiestos CSV estratificados (soporta --baseline)
│   │   ├── train_baselines.py # Entrypoint: entrena EfficientNet/MobileNet (baselines DL, funcional)
│   │   └── train.py           # Entrypoint: pipeline principal, loop de entrenamiento pendiente
│   └── vastai/
│       ├── onstart.sh         # Provisioning script para instancias vast.ai (clona, instala, descarga dataset)
│       └── launch.py          # Wrapper sobre la CLI `vastai`: search/create/run/sync/destroy
├── src/                       # Librería principal (instalable con pip install -e .)
│   ├── config.py               # PROJECT_ROOT, DATASET_ROOT y set_global_seed()
│   ├── analysis/
│   │   └── dataset_summary.py  # CLI: conteo de imágenes y tamaño por enfermedad/entorno
│   ├── cleanup/
│   │   └── find_duplicates.py  # CLI interactiva de deduplicación perceptual (PHash)
│   ├── data/
│   │   ├── loader.py           # load_and_normalize_image(): carga PIL + corrección EXIF/RGB
│   │   ├── dataset.py          # CornDataset (torch.Dataset) + build_weighted_sampler
│   │   ├── splitter.py         # HierarchicalStratifiedSplitter - división estratificada label+env
│   │   └── transforms.py       # CornTransformFactory + MINORITY_CLASSES
│   └── models/
│       ├── registry.py         # MODEL_REGISTRY (patrón Factory)
│       └── baselines/          # efficientnet.py, mobilenet.py - registran los modelos baseline
├── Dockerfile / .dockerignore  # Imagen reproducible (Python 3.11 + torch CUDA) para GPU remota
├── .env                        # DATASET_ROOT, HF_DATASET_REPO, HF_TOKEN, GDRIVE_DATASET_ID (no commiteado)
├── .env.example                # Plantilla de variables de entorno
├── pyproject.toml              # Dependencias (incl. extra [cloud]) y configuración de ruff
└── Makefile                    # Atajos de comandos frecuentes
```

---

## Arquitectura del código (`src/`)

### Capas y responsabilidades

| Módulo | Rol | Dependencias internas |
|---|---|---|
| `src/config.py` | Resuelve rutas de entorno (`DATASET_ROOT`, `PROJECT_ROOT`) y `set_global_seed()` | - |
| `src/data/loader.py` | Carga atómica de imagen con corrección EXIF y normalización RGB | `config.py` |
| `src/data/splitter.py` | División estratificada reproducible por `label + environment` | - |
| `src/data/transforms.py` | Factory de pipelines torchvision por etapa (`train`, `minority`, `val`, `test`) + `MINORITY_CLASSES` | `config.py` |
| `src/data/dataset.py` | `CornDataset` - `torch.Dataset` para DataLoader + `build_weighted_sampler` | `loader.py`, `transforms.py`, `config.py` |
| `src/models/registry.py` | `MODEL_REGISTRY` - registro de arquitecturas por nombre (Factory) | - |
| `src/models/baselines/` | `efficientnet_b0`, `efficientnet_lite0`, `mobilenet_v3_large` - registran modelos en `MODEL_REGISTRY` | `registry.py` |
| `src/analysis/dataset_summary.py` | CLI de conteo de imágenes por clase/entorno | `config.py` |
| `src/cleanup/find_duplicates.py` | CLI interactiva de deduplicación perceptual (PHash vía imagededup) | `config.py` |

### Principios de diseño

- **Punto de entrada único a imagen:** toda carga pasa por `load_and_normalize_image()`. Garantiza corrección EXIF y RGB antes de cualquier transformación.
- **Config centralizada:** `config/dataset.yaml` declara `dataset.classes` (las 9 clases completas), `target_size`, `seed`, y el perfil `baseline` (subset de clases + límite de imágenes por clase). Los módulos lo leen; nunca tienen constantes hardcodeadas de dominio.
- **Sin `sys.path.append` en código nuevo.** El paquete está instalado en modo editable (`pip install -e .`). Los imports `src.*` resuelven sin manipulación del path.
- **Clases minoritarias declaradas en `transforms.py`:** `MINORITY_CLASSES` es un `frozenset` estático (por nombre de clase, con los ratios de desbalance documentados). `CornDataset.__getitem__` lo consulta para aplicar el pipeline de augmentation extendido — es independiente del subconjunto de clases o del límite por clase que se haya usado al generar los splits.

---

## Pipelines

### Pipeline de datos compartido

```
clean/<clase>/{lab,real}/  ->  scripts/pipeline/create_splits.py
                            ->  splits/seed_42/{train,val,test}.csv            (9 clases, 70/15/15)
                            ->  splits/seed_42_baseline/{train,val,test}.csv   (perfil baseline, con --baseline)
```

`create_splits.py` valida integridad PIL, deduplica por SHA-256 y estratifica por `label + environment`
(`HierarchicalStratifiedSplitter`). Soporta:
- `--baseline`: usa el subset de clases y el límite de imágenes por clase definidos en
  `config/dataset.yaml -> baseline:` (por defecto: `healthy`, `common_rust`, `fall_armyworm`,
  `nitrogen_deficiency`, 500 imágenes por clase).
- `--classes <clase> [<clase> ...]` / `--max-per-class <N>`: overrides explícitos por CLI, tanto en
  modo `--baseline` como en modo completo.

### Pipeline de baselines (Deep Learning: EfficientNet / MobileNet)

Pese al nombre "baselines", **no es un pipeline sklearn** — es un pipeline PyTorch completo y
funcional, pensado como punto de comparación rápido y barato antes de comprometer el entrenamiento
completo del pipeline principal.

```
splits/seed_42_baseline/train.csv (o splits/seed_42/ para el dataset completo)
    -> CornDataset(transform=CornTrainingTransforms, minority_transform=CornMinorityTransforms)
    -> build_weighted_sampler (WeightedRandomSampler)
    -> DataLoader
    -> MODEL_REGISTRY.build(<efficientnet_b0|efficientnet_lite0|mobilenet_v3_large>)
    -> entrenamiento + evaluación (scripts/pipeline/train_baselines.py, funcional de punta a punta)
```

`make train-baselines` usa el perfil `baseline` (rápido, subset configurable); `make train-baselines-full`
usa el dataset completo (9 clases, sin límite por clase). Ambos aceptan `MODELS=<nombre>` para entrenar
un solo modelo en vez de todos (p.ej. `make train-baselines MODELS=efficientnet_b0`).

### Pipeline principal (Deep Learning)

```
splits/seed_42/train.csv
    -> CornDataset(transform=CornTrainingTransforms, minority_transform=CornMinorityTransforms)
    -> WeightedRandomSampler
    -> DataLoader
    -> modelo (pendiente: loop de entrenamiento en scripts/pipeline/train.py)
```

Comparte toda la infraestructura de datos/modelos con el pipeline de baselines; lo único que falta
implementar es el loop de entrenamiento en sí (`scripts/pipeline/train.py`).

---

## Clases del dataset

Definidas en `config/dataset.yaml -> dataset.classes`. Orden canónico para `class_to_idx`:

| Clase | Entornos disponibles | Nota |
|---|---|---|
| `common_rust` | lab, real | Minoritaria (ratio 3.9x) |
| `fall_armyworm` | real | - |
| `gray_leaf_spot` | lab, real | Minoritaria (ratio 7.9x) |
| `healthy` | lab, real | Clase mayoritaria de referencia |
| `lethal_necrosis` | real | - |
| `nitrogen_deficiency` | real | Minoritaria (ratio 16.8x) |
| `northern_corn_leaf_blight` | lab, real | - |
| `phosphorus_deficiency` | real | Minoritaria (ratio 14.3x) |
| `potassium_deficiency` | real | Minoritaria (ratio 32.9x) |

El perfil `baseline` (`config/dataset.yaml -> baseline.classes`) usa por defecto un subset de 4:
`healthy`, `common_rust`, `fall_armyworm`, `nitrogen_deficiency` (500 imágenes por clase).

---

## Dataset: hosting y descarga

`clean/` (~25k imágenes deduplicadas) se aloja como repo de tipo *dataset* en **Hugging Face Datasets
Hub** (fuente primaria) con **Google Drive** como respaldo. `scripts/dataset/download_dataset.py`
resuelve automáticamente cuál usar (`--source auto`, default). Variables relevantes en `.env`:
`HF_DATASET_REPO`, `HF_TOKEN` (opcional si ya hiciste `huggingface-cli login`), `GDRIVE_DATASET_ID`.

`scripts/download_datasets.sh` es un flujo distinto: ingesta de fuentes *crudas* nuevas
(Kaggle/Mendeley/Roboflow) hacia `raw/`, no toca `clean/`.

---

## Entrenamiento en GPU remota (vast.ai)

Ver guía completa en `docs/es/deployment/vast-ai.md`. Resumen: `Dockerfile` (Python 3.11 + PyTorch con
CUDA 12.1, instalado en `venv/` para que los mismos targets de `make` funcionen igual que en local) +
`scripts/vastai/onstart.sh` (provisioning script que clona el repo, arma el entorno y descarga el
dataset al arrancar la instancia) + `scripts/vastai/launch.py` (wrapper delgado sobre la CLI `vastai`
para buscar oferta, crear instancia, correr comandos por ssh, traer resultados y destruir la instancia).

---

## Comandos frecuentes (Makefile)

```bash
make install             # pip install -e ".[dev,analysis]"
make download-dataset    # Descarga clean/ (Hugging Face Hub, fallback Google Drive)
make splits              # Regenera splits CSV completos (9 clases) desde clean/
make splits-baseline     # Regenera splits del perfil baseline (subset de clases + límite por clase)
make train-baselines            # Entrena baselines sobre el perfil baseline (MODELS=<nombre> para uno solo)
make train-baselines-full       # Entrena baselines sobre el dataset completo
make train                # Ejecuta scripts/pipeline/train.py (loop de entrenamiento pendiente)
make summary               # Conteo de imágenes por clase/entorno
make lint / make fmt       # ruff check / ruff format
```

Comandos directos útiles:

```bash
# Deduplicación interactiva (requiere [analysis] extras)
python src/cleanup/find_duplicates.py

# Resumen detallado del dataset
python src/analysis/dataset_summary.py

# Splits con overrides explícitos (sin tocar config/dataset.yaml)
python scripts/pipeline/create_splits.py --classes healthy common_rust --max-per-class 200
```

---

## Configuración del entorno

1. Copiar `.env.example` a `.env` y ajustar `DATASET_ROOT` (y, si aplica, `HF_DATASET_REPO` /
   `GDRIVE_DATASET_ID` para `make download-dataset`).
2. `make install` instala el paquete en modo editable con dependencias dev.
3. Para análisis (FiftyOne, imagededup): `pip install -e ".[analysis]"`. Para subir/bajar el dataset
   desde Hugging Face/Drive: `pip install -e ".[cloud]"`.
4. El directorio `data/` en la raíz es un symlink a `DATASET_ROOT` (solo convención local; el código
   siempre resuelve rutas vía la variable de entorno `DATASET_ROOT`, no depende del symlink).

---
