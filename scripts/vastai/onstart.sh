#!/usr/bin/env bash
# Se ejecuta dentro de la instancia vast.ai al arrancar (pasado via `vastai create instance --onstart`).
# Clona el repo, instala el proyecto, arma el .env remoto y descarga el dataset limpio.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/daiv05/corn-leaf-desease-project.git}"
REPO_BRANCH="${REPO_BRANCH:-master}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace/corn-leaf-desease-project}"
REMOTE_DATASET_ROOT="${REMOTE_DATASET_ROOT:-/workspace/data}"

echo "[onstart] Clonando $REPO_URL ($REPO_BRANCH) en $WORKSPACE_DIR"
if [[ -d "$WORKSPACE_DIR/.git" ]]; then
  git -C "$WORKSPACE_DIR" pull
else
  git clone --branch "$REPO_BRANCH" "$REPO_URL" "$WORKSPACE_DIR"
fi
cd "$WORKSPACE_DIR"

# Se usa venv/bin/python (no el Python global) porque es lo que el Makefile del proyecto
# espera en Linux/macOS ($(PYTHON) := venv/bin/python) — así los mismos targets `make`
# funcionan igual en local y en la instancia. Si la imagen ya trae venv/ (p.ej. una imagen
# propia construida con el Dockerfile del repo) se reutiliza tal cual.
if [[ ! -x venv/bin/python ]]; then
  echo "[onstart] Creando venv/"
  python3 -m venv venv
fi
echo "[onstart] Instalando el proyecto (pip install -e .[cloud])"
venv/bin/pip install --no-cache-dir -e ".[cloud]"

echo "[onstart] Escribiendo .env remoto"
mkdir -p "$REMOTE_DATASET_ROOT"
cat > .env <<EOF
DATASET_ROOT=$REMOTE_DATASET_ROOT
HF_DATASET_REPO=${HF_DATASET_REPO:-}
HF_TOKEN=${HF_TOKEN:-}
GDRIVE_DATASET_ID=${GDRIVE_DATASET_ID:-}
EOF

echo "[onstart] Descargando dataset limpio"
venv/bin/python scripts/dataset/download_dataset.py

echo "[onstart] Listo. Conéctate por ssh y corre, por ejemplo:"
echo "  make splits-baseline && make train-baselines"
