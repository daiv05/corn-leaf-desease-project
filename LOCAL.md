# LOCAL.md - Levantar el proyecto en local

Guía paso a paso para dejar el proyecto corriendo en tu máquina: entorno virtual, variables de entorno, dependencias y descarga del dataset.

## Requisitos previos

- Python >= 3.11
- `make` disponible en el PATH (en Windows: Git Bash, WSL, o `choco install make`)

## 1. Clonar el repo

```bash
git clone https://github.com/daiv05/corn-leaf-desease-project
cd corn-leaf-desease-project
```

## 2. Crear el entorno virtual

El `Makefile` asume que el venv vive en `venv/` en la raíz del proyecto (detecta Windows vs. Linux/macOS automáticamente para elegir `venv/Scripts` o `venv/bin`).

```bash
python -m venv venv
```

Actívalo antes de correr cualquier comando fuera de `make` (los targets de `make` ya invocan el Python del venv directamente, sin necesidad de activarlo):

```bash
# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Windows (cmd)
venv\Scripts\activate.bat

# Linux/macOS
source venv/bin/activate
```

## 3. Configurar `.env`

Copia la plantilla y ajusta las variables:

```bash
cp .env.example .env
```

Variables relevantes:

| Variable | Descripción |
|---|---|
| `DATASET_ROOT` | Ruta local donde vivirá el dataset (debe contener `raw/`, `clean/`, `splits/`). Elige cualquier carpeta de tu máquina, p.ej. `C:/Users/tu_usuario/datasets/corn-leaf-diseases` en Windows o `/Users/tu_usuario/datasets/corn-leaf-diseases` en macOS/Linux. |
| `HF_DATASET_REPO` | Repo de tipo *dataset* en Hugging Face Hub que contiene `clean/` (fuente primaria de descarga). |
| `HF_TOKEN` | Solo necesario si el repo de HF es privado o no hiciste `huggingface-cli login`. |
| `GDRIVE_DATASET_ID` | ID de carpeta pública de Google Drive, usada como respaldo si falla la descarga desde HF. |

`DATASET_ROOT` no tiene que ser `data/` dentro del repo — puede apuntar a cualquier ruta. El directorio `data/` en la raíz del proyecto es opcionalmente un symlink de conveniencia hacia `DATASET_ROOT`; el código nunca depende de ese symlink, siempre resuelve rutas leyendo la variable de entorno `DATASET_ROOT` (ver `src/config.py`).

Si quieres ese symlink para navegar el dataset más fácilmente desde el editor:

```bash
# Windows (PowerShell, como administrador o con Developer Mode activado)
New-Item -ItemType SymbolicLink -Path data -Target "C:\ruta\a\tu\dataset"

# Linux/macOS
ln -s /ruta/a/tu/dataset data
```

## 4. Instalar dependencias

```bash
make install
```

Esto corre `pip install -e ".[dev,analysis]"` dentro del venv (instala el paquete `src/` en modo editable + extras de desarrollo y análisis). Extras disponibles en `pyproject.toml`:

- `dev`: ipykernel, jupyterlab, matplotlib, seaborn, ruff, pyright
- `analysis`: imagededup, fiftyone, imageio, mongoengine, motor (necesario para deduplicación y
  exploración visual)
- `cloud`: huggingface_hub, gdown (necesario para descargar/subir el dataset)

Si solo necesitas descargar el dataset sin las herramientas de análisis:

```bash
venv\Scripts\pip install -e ".[cloud]"   # Windows
venv/bin/pip install -e ".[cloud]"        # Linux/macOS
```

## 5. Descargar el dataset (`clean/`)

Con `DATASET_ROOT` ya configurado en `.env` y las dependencias de `cloud` instaladas:

```bash
make download-dataset
```

Esto ejecuta `scripts/dataset/download_dataset.py`, que descarga `clean/` hacia
`$DATASET_ROOT/clean/`, intentando primero Hugging Face Hub (`HF_DATASET_REPO`) y usando Google Drive (`GDRIVE_DATASET_ID`) como respaldo si falla. Si `$DATASET_ROOT/clean/` ya tiene contenido, el script no vuelve a descargar (usa `--force` para forzarlo).

**Nunca coloques ni modifiques nada manualmente en `raw/`** — esa carpeta es inmutable y no forma parte de este flujo de descarga; `clean/` es la única fuente de verdad para el pipeline.

## 6. Generar los splits

Con `clean/` ya poblado:

```bash
make splits              # Splits completos (9 clases) -> splits/seed_42/
make splits-baseline      # Splits del perfil baseline (subset + límite por clase) -> splits/seed_42_baseline/
```

No edites los CSV de `splits/` a mano — son derivados reproducibles.

## 7. Verificar que todo funciona

```bash
make summary   # Conteo de imágenes por clase/entorno (valida que clean/ esté bien poblado)
make lint      # ruff check
make fmt       # ruff format
```

## 8. Entrenar

```bash
make train-baselines               # Entrena baselines (EfficientNet/MobileNet) sobre el perfil baseline
make train-baselines MODELS=efficientnet_b0   # Solo un modelo
make train-baselines-full          # Baselines sobre el dataset completo (9 clases)
make train                          # Pipeline principal (loop de entrenamiento aún pendiente)
```

## Resumen rápido (happy path)

```bash
git clone https://github.com/daiv05/corn-leaf-desease-project
cd corn-leaf-desease-project
python -m venv venv
source venv/bin/activate   # o venv\Scripts\Activate.ps1 en Windows
cp .env.example .env       # editar DATASET_ROOT / HF_DATASET_REPO / GDRIVE_DATASET_ID
make install
make download-dataset
make splits
make splits-baseline
make summary
```
