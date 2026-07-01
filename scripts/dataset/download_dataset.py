import argparse
import logging
import os
from pathlib import Path

from src.config import DATASET_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _clean_dir_has_content(clean_dir: Path) -> bool:
    return clean_dir.is_dir() and any(clean_dir.iterdir())


def _download_from_hf(repo_id: str, clean_dir: Path, token: str | None) -> None:
    from huggingface_hub import snapshot_download

    logger.info(f"Descargando desde Hugging Face Datasets Hub: {repo_id}")
    clean_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(clean_dir),
        token=token,
    )
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
    clean_dir = DATASET_ROOT / "clean"
    hf_repo = hf_repo or os.getenv("HF_DATASET_REPO")
    hf_token = hf_token or os.getenv("HF_TOKEN")
    gdrive_id = gdrive_id or os.getenv("GDRIVE_DATASET_ID")

    if not force and _clean_dir_has_content(clean_dir) and not dry_run:
        logger.info(
            f"{clean_dir} ya tiene contenido; se omite la descarga (usa --force para reintentar)."
        )
        return

    if source in ("hf", "auto") and not hf_repo and source == "hf":
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
