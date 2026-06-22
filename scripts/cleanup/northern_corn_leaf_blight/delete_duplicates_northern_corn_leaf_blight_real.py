"""
Elimina imagenes duplicadas en:
  data/clean/northern_corn_leaf_blight/real
Mantiene una copia por contenido (hash) y borra el resto.
"""

import hashlib
import os
from typing import Callable, Dict

try:
    import xxhash  # type: ignore
except Exception:
    xxhash = None

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SOURCE_DIR = os.path.join(
    BASE_DIR, "data", "clean", "northern_corn_leaf_blight", "real", "temporal"
)

CHUNK_SIZE = 1024 * 1024  # 1 MB


def resolve_hasher() -> tuple[Callable[[], object], str]:
    if xxhash is not None:
        return (lambda: xxhash.xxh64()), "xxhash.xxh64"
    return (lambda: hashlib.sha256()), "hashlib.sha256"


def file_hash(path: str, hasher_factory: Callable[[], object]) -> str:
    hasher = hasher_factory()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def delete_duplicates(source_dir: str) -> None:
    source_dir = os.path.abspath(source_dir)
    if not os.path.isdir(source_dir):
        print(f"Directorio no encontrado: {source_dir}")
        return

    entries = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    if not entries:
        print("No se encontraron archivos en el directorio.")
        return

    hasher_factory, hasher_label = resolve_hasher()
    print(f"Hash en uso: {hasher_label}")

    seen_hashes: Dict[str, str] = {}
    removed = 0
    skipped = 0

    for filename in entries:
        path = os.path.join(source_dir, filename)
        try:
            digest = file_hash(path, hasher_factory)
        except OSError:
            skipped += 1
            continue

        if digest in seen_hashes:
            os.remove(path)
            removed += 1
        else:
            seen_hashes[digest] = filename

    print(f"Duplicados eliminados: {removed}")
    if skipped:
        print(f"Omitidos (errores de lectura): {skipped}")


if __name__ == "__main__":
    delete_duplicates(SOURCE_DIR)
