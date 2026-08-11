import argparse
import logging
import os
import shutil
import tarfile
from pathlib import Path

import yaml

from src.config import PROJECT_ROOT, get_dataset_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _clean_dir_has_content(clean_dir: Path) -> bool:
    """True solo si existe al menos un directorio de clase del YAML con archivos
    (así una descarga interrumpida no bloquea el reintento automático).

    Un .tar pendiente cuenta como descarga incompleta: el proceso se cortó entre bajar los
    shards y extraerlos, dejando el árbol a medias. El reintento es idempotente.
    """
    if not clean_dir.is_dir():
        return False
    if any(clean_dir.rglob("*.tar")):
        return False
    with open(PROJECT_ROOT / "config" / "dataset.yaml") as f:
        classes = yaml.safe_load(f)["dataset"]["classes"]
    return any((clean_dir / c).is_dir() and any((clean_dir / c).iterdir()) for c in classes)


def _extract_and_remove_tars(clean_dir: Path) -> None:
    """Descomprime cada .tar en su lugar y lo borra al terminar"""
    tar_paths = sorted(clean_dir.rglob("*.tar"))
    for tar_path in tar_paths:
        logger.info(f"Extrayendo {tar_path.name}...")
        with tarfile.open(tar_path) as tf:
            tf.extractall(clean_dir, filter="data")
        tar_path.unlink()
        logger.info(f"{tar_path.name} extraído y eliminado.")


def _download_from_hf(repo_id: str, clean_dir: Path, token: str | None) -> None:
    from huggingface_hub import snapshot_download

    logger.info(f"Descargando desde Hugging Face Datasets Hub: {repo_id}")
    clean_dir.mkdir(parents=True, exist_ok=True)
    # La metadata solo existe para que HF reconozca el repo como dataset de imágenes; el
    # pipeline saca label/environment del árbol extraído, así que no se descarga.
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(clean_dir),
        token=token,
        ignore_patterns=[
            ".gitattributes",
            "README.md",
            "metadata.csv",
            "dataset_infos.json",
        ],
    )
    _extract_and_remove_tars(clean_dir)
    # Elimina los metadatos de descarga que snapshot_download deja en clean/.cache/
    shutil.rmtree(clean_dir / ".cache", ignore_errors=True)
    logger.info(f"Dataset descargado en {clean_dir}")


def _download_from_gdrive(gdrive_id: str, clean_dir: Path) -> None:
    import gdown

    logger.info(f"Descargando desde Google Drive (fallback): {gdrive_id}")
    clean_dir.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(id=gdrive_id, output=str(clean_dir), quiet=False, use_cookies=False)
    logger.info(f"Dataset descargado en {clean_dir}")


def download_clean_dataset(
    source: str = "auto",
    force: bool = False,
    hf_repo: str | None = None,
    hf_token: str | None = None,
    gdrive_id: str | None = None,
    dry_run: bool = False,
) -> None:
    clean_dir = get_dataset_root() / "clean"
    hf_repo = hf_repo or os.getenv("HF_DATASET_REPO")
    hf_token = hf_token or os.getenv("HF_TOKEN")
    gdrive_id = gdrive_id or os.getenv("GDRIVE_DATASET_ID")

    if not force and _clean_dir_has_content(clean_dir) and not dry_run:
        logger.info(
            f"{clean_dir} ya tiene contenido; se omite la descarga (usa --force para reintentar)."
        )
        return

    if source == "hf" and not hf_repo:
        raise SystemExit("--source hf requiere HF_DATASET_REPO (env) o --hf-repo.")
    if source == "gdrive" and not gdrive_id:
        raise SystemExit("--source gdrive requiere GDRIVE_DATASET_ID (env) o --gdrive-id.")
    if source == "auto" and not hf_repo and not gdrive_id:
        raise SystemExit(
            "No hay fuente configurada: define HF_DATASET_REPO o GDRIVE_DATASET_ID en .env "
            "(o pasa --hf-repo/--gdrive-id)."
        )

    if dry_run:
        plan = f"source={source} hf_repo={hf_repo!r} gdrive_id={gdrive_id!r} -> {clean_dir}"
        logger.info(f"[dry-run] Resolución de fuente válida: {plan}")
        return

    if source == "hf":
        _download_from_hf(hf_repo, clean_dir, token=hf_token)
        return
    if source == "gdrive":
        _download_from_gdrive(gdrive_id, clean_dir)
        return

    # auto: intenta HF primero, cae a Google Drive
    if hf_repo:
        try:
            _download_from_hf(hf_repo, clean_dir, token=hf_token)
            return
        except Exception as e:
            logger.warning(f"Descarga desde Hugging Face falló ({e}); probando Google Drive.")
    if gdrive_id:
        _download_from_gdrive(gdrive_id, clean_dir)
        return
    raise SystemExit("Descarga desde Hugging Face falló y no hay GDRIVE_DATASET_ID de fallback.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga el dataset limpio (clean/) hacia $DATASET_ROOT/clean/."
    )
    parser.add_argument(
        "--source",
        choices=["hf", "gdrive", "auto"],
        default="auto",
        help="Fuente a usar. 'auto' intenta Hugging Face y cae a Google Drive (default).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Vuelve a descargar aunque clean/ ya tenga contenido."
    )
    parser.add_argument(
        "--hf-repo", dest="hf_repo", default=None, help="Override de HF_DATASET_REPO."
    )
    parser.add_argument("--hf-token", dest="hf_token", default=None, help="Override de HF_TOKEN.")
    parser.add_argument(
        "--gdrive-id", dest="gdrive_id", default=None, help="Override de GDRIVE_DATASET_ID."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo valida qué fuente se usaría, sin descargar nada.",
    )
    args = parser.parse_args()

    download_clean_dataset(
        source=args.source,
        force=args.force,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token,
        gdrive_id=args.gdrive_id,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
