# CLAUDE.md — Corn Leaf Disease Project

## Reglas de datos

- **Nunca modificar `raw/`.** La carpeta `$DATASET_ROOT/raw` es inmutable y solo sirve como fuente original.
- `clean/` es la única fuente de verdad para el pipeline de entrenamiento. Estructura fija: `clean/<clase>/{lab,real}/`.
- Los splits CSV (`splits/seed_42/`) son derivados reproducibles: se regeneran con `make splits`. No editarlos a mano.

---

## Estructura del proyecto

```
corn-leaf-desease-project/
├── config/
│   └── dataset.yaml          # Parámetros centralizados: clases, tamaños, seeds, features
├── docs/                     # Documentación VitePress (es/)
├── fiftyone/                 # Scripts de exploración visual con FiftyOne
├── notebooks/                # EDA en Jupyter y otras notebooks de análisis
├── scripts/
│   ├── cleanup/              # Scripts de limpieza por clase/dataset (one-shot, ya ejecutados)
│   │   ├── common_rust/
│   │   ├── fall_armyworm/
│   │   ├── gray_leaf_spot/
│   │   ├── healthy/
│   │   ├── nitrogen_deficiency/
│   │   ├── northern_corn_leaf_blight/
│   │   ├── others/
│   │   ├── phosphorus_deficiency/
│   │   └── potassium_deficiency/
│   └── pipeline/
│       ├── create_splits.py  # Entrypoint: genera manifiestos CSV estratificados
│       └── train.py          # Entrypoint: construye DataLoaders y criterio (loop pendiente)
├── src/                      # Librería principal (instalable con pip install -e .)
│   ├── config.py             # PROJECT_ROOT y DATASET_ROOT resueltos desde .env
│   ├── cleanup/
│   │   └── find_duplicates.py  # CLI interactiva de deduplicación perceptual (PHash)
│   ├── data/
│   │   ├── loader.py         # load_and_normalize_image(): carga PIL + corrección EXIF/RGB
│   │   ├── dataset.py        # CornDataset (torch.Dataset) — consumo indexado por DataLoader
│   │   ├── feature_dataset.py # CornFeatureDataset — devuelve (np.ndarray, int) para sklearn
│   │   ├── splitter.py       # HierarchicalStratifiedSplitter — división estratificada label+env
│   │   ├── transforms.py     # CornTransformFactory + pipelines train/minority/val
│   │   ├── dataset_summary.py # CLI: conteo de imágenes y tamaño por enfermedad/entorno
│   │   └── count_dataset.py  # Función auxiliar de conteo (Markdown output)
│   └── features/
│       └── extractors.py     # HOGExtractor, HSVHistogramExtractor, LBPExtractor,
│                             # GLCMExtractor, CombinedExtractor (baselines)
├── .env                      # DATASET_ROOT (no commiteado)
├── .env.example              # Plantilla de variables de entorno
├── pyproject.toml            # Dependencias y configuración de ruff
└── Makefile                  # Atajos de comandos frecuentes
```

---

## Arquitectura del código (`src/`)

### Capas y responsabilidades

| Módulo | Rol | Dependencias internas |
|---|---|---|
| `src/config.py` | Resuelve rutas de entorno (`DATASET_ROOT`, `PROJECT_ROOT`) | — |
| `src/data/loader.py` | Carga atómica de imagen con corrección EXIF y normalización RGB | `config.py` |
| `src/data/splitter.py` | División estratificada reproducible por `label + environment` | — |
| `src/data/transforms.py` | Factory de pipelines torchvision por etapa (`train`, `minority`, `val`, `test`) | `config.py` |
| `src/data/dataset.py` | `CornDataset` — `torch.Dataset` para DataLoader, usa transforms torchvision | `loader.py`, `transforms.py`, `config.py` |
| `src/data/feature_dataset.py` | `CornFeatureDataset` — dataset para baselines sklearn, devuelve `np.ndarray` | `loader.py`, `features/extractors.py`, `config.py` |
| `src/features/extractors.py` | `HOG`, `HSV`, `LBP`, `GLCM`, `CombinedExtractor` — vectores de features | `config.py` |
| `src/cleanup/find_duplicates.py` | CLI interactiva de deduplicación perceptual (PHash vía imagededup) | `config.py` |

### Principios de diseño

- **Punto de entrada único a imagen:** toda carga pasa por `load_and_normalize_image()`. Garantiza corrección EXIF y RGB antes de cualquier transformación.
- **Config centralizada:** `config/dataset.yaml` declara clases, `target_size`, `seed` y parámetros de features. Los módulos lo leen; nunca tienen constantes hardcodeadas de dominio.
- **Sin `sys.path.append` en código nuevo.** El paquete está instalado en modo editable (`pip install -e .`). Los imports `src.*` resuelven sin manipulación del path.
- **Dos datasets paralelos, sin mezclar tipos:** `CornDataset` devuelve `(torch.Tensor, int)`, `CornFeatureDataset` devuelve `(np.ndarray, int)`. No se adaptan entre sí.
- **Clases minoritarias declaradas en `transforms.py`:** `MINORITY_CLASSES` es un `frozenset` con los ratios de desbalance documentados. `CornDataset.__getitem__` lo consulta para aplicar el pipeline de augmentation extendido.

---

## Pipelines

### Pipeline de datos compartido

```
clean/<clase>/{lab,real}/  →  scripts/pipeline/create_splits.py
                           →  splits/seed_42/{train,val,test}.csv   (70/15/15)
```

`create_splits.py` valida integridad PIL, deduplica por SHA-256 y estratifica por `label + environment`.

### Pipeline Deep Learning (PyTorch)

```
splits/seed_42/train.csv
    → CornDataset(transform=CornTrainingTransforms, minority_transform=CornMinorityTransforms)
    → WeightedRandomSampler
    → DataLoader
    → modelo (pendiente: loop de entrenamiento en scripts/pipeline/train.py)
```

### Pipeline Baselines (sklearn)

```
splits/seed_42/train.csv
    → CornFeatureDataset(extractor=CombinedExtractor.from_config())
    → CornFeatureDataset.load_all()  →  (X: np.ndarray, y: np.ndarray)
    → StandardScaler + SVM / RandomForest / k-NN  (class_weight='balanced')
    → entrypoint pendiente: scripts/pipeline/train_baselines.py
```

---

## Clases del dataset

Definidas en `config/dataset.yaml → dataset.classes`. Orden canónico para `class_to_idx`:

| Clase | Entornos disponibles | Nota |
|---|---|---|
| `aphids_pest` | real | Excluida del entrenamiento actual (`EXCLUDE_CLASSES`) |
| `common_rust` | lab, real | Minoritaria (ratio 3.9x) |
| `fall_armyworm` | real | — |
| `gray_leaf_spot` | lab, real | Minoritaria (ratio 7.9x) |
| `healthy` | lab, real | Clase mayoritaria de referencia |
| `nitrogen_deficiency` | real | Minoritaria (ratio 16.8x) |
| `northern_corn_leaf_blight` | lab, real | — |
| `phosphorus_deficiency` | real | Minoritaria (ratio 14.3x) |
| `potassium_deficiency` | real | Minoritaria (ratio 32.9x) |

---

## Comandos frecuentes (Makefile)

```bash
make install        # pip install -e ".[dev]"
make splits         # Regenera splits CSV desde clean/
make train          # Ejecuta scripts/pipeline/train.py
make summary        # Conteo de imágenes por clase/entorno
make lint           # ruff check
make fmt            # ruff format
```

Comandos directos útiles:

```bash
# Deduplicación interactiva (requiere [analysis] extras)
python src/cleanup/find_duplicates.py

# Resumen detallado del dataset
python src/data/dataset_summary.py
```

---

## Configuración del entorno

1. Copiar `.env.example` a `.env` y ajustar `DATASET_ROOT`.
2. `make install` instala el paquete en modo editable con dependencias dev.
3. Para análisis (FiftyOne, imagededup): `pip install -e ".[analysis]"`.
4. El directorio `data/` en la raíz es un symlink a `DATASET_ROOT`.

---
