"""
Mueve y renombra las imagenes en:
    data/clean/fall_armyworm/multi_desease/  (busqueda recursiva)
hacia:
    data/clean/fall_armyworm/real/
con el patron:
    fall_armyworm_multi_desease_real_<randomNumber>.<ext>
"""

import os
import sys
import random
import shutil
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.config import DATASET_ROOT

BASE_DIR = DATASET_ROOT / "clean" / "fall_armyworm"
SOURCE_DIR = BASE_DIR / "multi_desease"
DEST_DIR = BASE_DIR / "real"

RANDOM_DIGITS = 8
PREFIX = "fall_armyworm_multi_desease_real_"


def collect_images(source_dir: str) -> list[str]:
    images = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            ext = os.path.splitext(f)[1]
            if ext:
                images.append(os.path.join(root, f))
    return images


def move_and_rename(source_dir: str, dest_dir: str) -> None:
    source_dir = os.path.abspath(source_dir)
    dest_dir = os.path.abspath(dest_dir)

    if not os.path.isdir(source_dir):
        print(f"ERROR: directorio fuente no existe: {source_dir}")
        return

    os.makedirs(dest_dir, exist_ok=True)

    files = collect_images(source_dir)
    if not files:
        print("No se encontraron archivos en el directorio fuente.")
        return

    print(f"Archivos encontrados: {len(files)}")
    print(f"Destino: {dest_dir}")
    confirm = input("Continuar? [s/N]: ").strip().lower()
    if confirm != "s":
        print("Operacion cancelada.")
        return

    used_numbers: set[int] = set()
    moved = 0
    skipped = 0

    for src in files:
        ext = os.path.splitext(src)[1].lower()
        if not ext:
            skipped += 1
            continue

        while True:
            number = random.randint(10 ** (RANDOM_DIGITS - 1), 10**RANDOM_DIGITS - 1)
            new_name = f"{PREFIX}{number}{ext}"
            dst = os.path.join(dest_dir, new_name)
            if number not in used_numbers and not os.path.exists(dst):
                used_numbers.add(number)
                break

        shutil.move(src, dst)
        moved += 1

    print(f"Movidas y renombradas: {moved} imagenes")
    if skipped:
        print(f"Omitidas (sin extension): {skipped}")


if __name__ == "__main__":
    move_and_rename(SOURCE_DIR, DEST_DIR)
