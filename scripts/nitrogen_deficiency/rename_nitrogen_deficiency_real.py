"""
Renombra las imágenes en:
  data/clean/nitrogen_deficiency/real/
con el patrón:
  nitrogen_deficiency_maize_nutrient_real_<número_aleatorio_único>.<ext>
"""

import os
import random

SOURCE_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "data", "clean", "nitrogen_deficiency", "real"
)

PREFIX = "nitrogen_deficiency_maize_nutrient_real_"
RANDOM_DIGITS = 8


def rename_images(source_dir: str) -> None:
    source_dir = os.path.abspath(source_dir)

    entries = [
        f for f in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, f))
    ]

    if not entries:
        print("No se encontraron archivos en el directorio fuente.")
        return

    used_numbers: set[int] = set()
    renamed = 0
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
        dst = os.path.join(source_dir, new_name)

        os.rename(src, dst)
        renamed += 1

    print(f"Renombradas: {renamed} imágenes")
    print(f"Directorio: {source_dir}")
    if skipped:
        print(f"Omitidas (sin extensión): {skipped}")


if __name__ == "__main__":
    rename_images(SOURCE_DIR)
