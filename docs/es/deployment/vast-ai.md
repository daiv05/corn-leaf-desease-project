# Entrenamiento reproducible en GPU (vast.ai)

Esta guía cubre cómo correr el pipeline de baselines (`train_baselines.py`) o el pipeline
principal (`train.py`) exactamente igual en local (CPU) y en una instancia GPU alquilada en
[vast.ai](https://vast.ai). El único cambio entre ambos entornos es el contenido de `.env`
(`DATASET_ROOT`) — el código ya selecciona `cuda` automáticamente si está disponible
(`torch.cuda.is_available()`).

## Flujo, local vs. vast.ai

| Paso | Local (Windows/CPU) | Instancia vast.ai (GPU) |
|---|---|---|
| Código | `git clone` + editar en el equipo | `scripts/vastai/onstart.sh` clona el repo dentro de la instancia |
| `.env` | `DATASET_ROOT=C:/Users/tu_usuario/.../data` | `DATASET_ROOT=/workspace/data` (lo escribe `onstart.sh`) |
| Instalación | `make install` (venv creado a mano) | `venv/bin/pip install -e ".[cloud]"` (venv creado por `onstart.sh`) |
| Dataset | `make download-dataset` | `python scripts/dataset/download_dataset.py` (dentro de `onstart.sh`) |
| Splits | `make splits-baseline` / `make splits` | igual, por ssh |
| Entrenamiento | `make train-baselines` / `make train` | igual, por ssh |
| Resultados | quedan en `$DATASET_ROOT/results/` | se traen con `vastai copy` / `scripts/vastai/launch.py sync` |

## 1. Dataset: Hugging Face Datasets Hub (fuente primaria) + Google Drive (fallback)

El dataset limpio (`clean/<clase>/{lab,real}/`, ~25k imágenes deduplicadas) se aloja como
repo de tipo *dataset* en Hugging Face Hub — descargas más rápidas y estables desde una
instancia efímera que un enlace de Google Drive (sin límites de tamaño ni interstitial de
virus scan). El enlace de Google Drive existente se mantiene como respaldo.

- Subir (una vez, mantenedor): `python scripts/dataset/upload_to_hf.py --repo-id usuario/corn-leaf-clean`
  (requiere `huggingface-cli login` o `HF_TOKEN` en `.env`).
- Descargar (local o remoto): `make download-dataset`, que corre
  `scripts/dataset/download_dataset.py --source auto` — intenta Hugging Face
  (`HF_DATASET_REPO`) primero y cae a Google Drive (`GDRIVE_DATASET_ID`) si falla o no está
  configurado. Es idempotente: si `clean/` ya tiene contenido, no vuelve a descargar salvo `--force`.

Variables relevantes en `.env` (ver `.env.example`): `HF_DATASET_REPO`, `HF_TOKEN`,
`GDRIVE_DATASET_ID`.

## 2. Imagen reproducible (Docker)

El `Dockerfile` de la raíz usa `python:3.11-slim` (el proyecto requiere Python >= 3.11,
ver `pyproject.toml`) + wheels de PyTorch con CUDA 12.1 desde el índice oficial de PyTorch,
e instala el proyecto en un `venv/` (no en el Python global) para que los mismos targets de
`make` funcionen igual que en local. Sirve para verificar en el propio equipo
(`docker build -t corn-leaf-baselines .`) que el entorno instala limpio antes de tocar una
instancia real.

Para vast.ai no hace falta construir ni publicar esta imagen: `scripts/vastai/launch.py`
usa por defecto la plantilla oficial `vastai/pytorch:2.6.0-cuda-12.6.3-py312` (Python 3.12,
también satisface `>=3.11`) y `scripts/vastai/onstart.sh` clona el repo y arma el mismo
`venv/` en caliente al arrancar la instancia. Ojo: las imágenes oficiales
`pytorch/pytorch:*-runtime` traen Python 3.10 vía conda y **no** sirven para este proyecto
(la instalación falla por el requisito de Python en `pyproject.toml`).

## 3. Lanzar una instancia en vast.ai

Requiere `pip install vastai` y `vastai set api-key <tu-api-key>` ya configurados.

```bash
# 1. Buscar una oferta de GPU disponible (ajusta el filtro a tu presupuesto/GPU deseada)
python scripts/vastai/launch.py search 'gpu_name=RTX_3090 num_gpus=1 dph<0.30' -o 'dph+'

# 2. Crear la instancia (usa scripts/vastai/onstart.sh como provisioning script)
python scripts/vastai/launch.py create <OFFER_ID> \
  --env HF_DATASET_REPO=usuario/corn-leaf-clean \
  --env HF_TOKEN=hf_xxx

# 3. Correr el pipeline por ssh (una vez el onstart terminó de descargar el dataset)
python scripts/vastai/launch.py run <INSTANCE_ID> make splits-baseline
python scripts/vastai/launch.py run <INSTANCE_ID> make train-baselines
# o el pipeline principal:
python scripts/vastai/launch.py run <INSTANCE_ID> make train

# 4. Traer resultados de vuelta
python scripts/vastai/launch.py sync <INSTANCE_ID>

# 5. Destruir la instancia (vast.ai cobra por minuto mientras esté viva)
python scripts/vastai/launch.py destroy <INSTANCE_ID>
```

Cada subcomando acepta `--dry-run` para ver el comando `vastai`/`ssh` exacto sin ejecutarlo.

## 4. Alcance de `train.py`

`train.py` (pipeline principal) todavía tiene pendiente el loop de entrenamiento — construye
modelo, datasets y criterio, pero no entrena. Esta guía deja el entorno (Docker, dataset,
`.env`, orquestación vast.ai) listo para ese pipeline también; implementar el loop es un
trabajo aparte.
