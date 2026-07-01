import argparse
import logging
import os

from src.config import DATASET_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def upload_clean_dataset(repo_id: str, token: str | None = None, private: bool = False) -> None:
    """Sube $DATASET_ROOT/clean/ tal cual (mismo árbol <clase>/{lab,real}/) a un repo de
    tipo dataset en Hugging Face Hub. No reestructura a formato `imagefolder`: los splits
    train/val/test se generan localmente con create_splits.py, HF solo necesita alojar
    los archivos fuente.
    """
    from huggingface_hub import HfApi

    clean_dir = DATASET_ROOT / "clean"
    if not clean_dir.is_dir():
        raise SystemExit(f"No se encontró {clean_dir}. Verifica DATASET_ROOT en .env")

    api = HfApi(token=token)
    logger.info(f"Creando/verificando repo de dataset: {repo_id} (private={private})")
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    logger.info(f"Subiendo {clean_dir} -> hf://datasets/{repo_id}")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(clean_dir),
        commit_message="Actualiza dataset clean/",
    )
    logger.info("Subida completada.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sube el dataset limpio (clean/) a Hugging Face Datasets Hub."
    )
    parser.add_argument(
        "--repo-id",
        default=os.getenv("HF_DATASET_REPO"),
        help="Repo destino, formato 'usuario/nombre-dataset'. Default: env HF_DATASET_REPO.",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="Token de HF. Si se omite, usa HF_TOKEN o la sesión de `huggingface-cli login`.",
    )
    parser.add_argument("--private", action="store_true", help="Crea el repo como privado.")
    args = parser.parse_args()

    if not args.repo_id:
        raise SystemExit(
            "Falta --repo-id (o define HF_DATASET_REPO en .env), formato 'usuario/nombre-dataset'."
        )

    upload_clean_dataset(repo_id=args.repo_id, token=args.token, private=args.private)


if __name__ == "__main__":
    main()
