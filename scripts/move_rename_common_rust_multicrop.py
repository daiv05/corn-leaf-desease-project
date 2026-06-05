"""
Renombra y mueve las imágenes desde:
  data/clean/common-rust/lab/multicrop/
hacia:
  data/clean/common-rust/lab/
con el patrón:
  common_rust_multi_desease_lab_<número_aleatorio_único>.<ext>
"""

import os
import random
import shutil

SOURCE_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "data", "clean", "common-rust", "lab", "multicrop"
)
DEST_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "data", "clean", "common-rust", "lab"
)

PREFIX = "common_rust_multi_desease_lab_"
RANDOM_DIGITS = 8


def move_and_rename(source_dir: str, dest_dir: str) -> None:
    source_dir = os.path.abspath(source_dir)
    dest_dir = os.path.abspath(dest_dir)

    entries = [
        f for f in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, f))
    ]

    if not entries:
        print("No se encontraron archivos en el directorio fuente.")
        return

    used_numbers: set[int] = set()
    moved = 0
    skipped = 0

    for filename in entries:
        ext = os.path.splitext(filename)[1].lower()
        if not ext:
            skipped += 1
            continue

        while True:
            number = random.randint(10 ** (RANDOM_DIGITS - 1), 10**RANDOM_DIGITS - 1)
            if number not in used_numbers:
                used_numbers.add(number)
                break

        new_name = f"{PREFIX}{number}{ext}"
        src = os.path.join(source_dir, filename)
        dst = os.path.join(dest_dir, new_name)

        shutil.move(src, dst)
        moved += 1

    print(f"Movidas y renombradas: {moved} imágenes")
    print(f"Destino: {dest_dir}")
    if skipped:
        print(f"Omitidas (sin extensión): {skipped}")


if __name__ == "__main__":
    move_and_rename(SOURCE_DIR, DEST_DIR)
