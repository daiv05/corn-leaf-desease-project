"""
Mueve las imagenes desde:
  data/clean/gray_leaf_spot/real
hacia:
  data/clean/gray_leaf_spot/real
solo cuando el nombre contiene:
  gray_leaf_spot_cropdg_real_
sin cambiar el nombre de archivo.
"""

import os
import shutil

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SOURCE_DIR = os.path.join(
    BASE_DIR,
    "data", "clean", "gray_leaf_spot", "real"
)
DEST_DIR = os.path.join(
    BASE_DIR,
    "data", "clean", "gray_leaf_spot", "lab"
)

NAME_PATTERN = "gray_leaf_spot_cropdg_real_"


def move_images(source_dir: str, dest_dir: str, name_pattern: str) -> None:
    source_dir = os.path.abspath(source_dir)
    dest_dir = os.path.abspath(dest_dir)

    if not os.path.isdir(source_dir):
        print(f"Directorio no encontrado: {source_dir}")
        return

    os.makedirs(dest_dir, exist_ok=True)

    entries = [
        f for f in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, f)) and name_pattern in f
    ]

    if not entries:
        print("No se encontraron archivos que coincidan en el directorio fuente.")
        return

    moved = 0
    skipped = 0

    for filename in entries:
        src = os.path.join(source_dir, filename)
        dst = os.path.join(dest_dir, filename)

        if os.path.exists(dst):
            skipped += 1
            continue

        shutil.move(src, dst)
        moved += 1

    print(f"Movidas: {moved} imagenes")
    print(f"Destino: {dest_dir}")
    if skipped:
        print(f"Omitidas (ya existian en destino): {skipped}")


if __name__ == "__main__":
    move_images(SOURCE_DIR, DEST_DIR, NAME_PATTERN)