"""
Renombra las imagenes en:
    data/clean/gray_leaf_spot/real
al patron:
    gray_leaf_spot_<origen>_<tipo>_<correlativo>.<ext>
con correlativo iniciando en 1.
"""

import os
import random

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SOURCE_DIR = os.path.join(BASE_DIR, "data", "clean", "gray_leaf_spot", "real", "temporal")

START_INDEX = 1
RANDOM_DIGITS = 8


def ask_source_abbr() -> str:
    while True:
        value = input("Abreviatura de dataset de origen: ").strip()
        if value:
            return value

        print("Abreviatura requerida.")


def ask_image_type() -> str:
    while True:
        choice = input("Tipo [1=real,2=lab]: ").strip()
        if choice == "1":
            return "real"
        if choice == "2":
            return "lab"

        print("Elige 1 o 2.")


def ask_order_mode() -> str:
    while True:
        choice = input("Orden [1=aleatorio,2=secuencial]: ").strip()
        if choice == "1":
            return "random"
        if choice == "2":
            return "sequential"

        print("Elige 1 o 2.")


def rename_images(source_dir: str, prefix: str, order_mode: str) -> None:
    source_dir = os.path.abspath(source_dir)
    entries = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    if not entries:
        print("No se encontraron archivos en el directorio.")
        return

    if order_mode == "sequential":
        entries.sort()
    else:
        random.shuffle(entries)

    renamed = 0
    skipped = 0
    index = START_INDEX

    used_numbers: set[int] = set()

    for filename in entries:
        ext = os.path.splitext(filename)[1].lower()
        if not ext:
            skipped += 1
            continue

        if order_mode == "sequential":
            number = index
            index += 1
        else:
            # generate a unique random 8-digit number (10000000-99999999)
            while True:
                number = random.randint(10 ** (RANDOM_DIGITS - 1), 10**RANDOM_DIGITS - 1)
                new_name = f"{prefix}{number}{ext}"
                dst = os.path.join(source_dir, new_name)
                if number not in used_numbers and not os.path.exists(dst):
                    used_numbers.add(number)
                    break

        new_name = f"{prefix}{number}{ext}"
        src = os.path.join(source_dir, filename)
        dst = os.path.join(source_dir, new_name)

        os.rename(src, dst)
        renamed += 1

    print(f"Renombradas: {renamed} imagenes")
    if skipped:
        print(f"Omitidas (sin extension): {skipped}")


if __name__ == "__main__":
    source_abbr = ask_source_abbr()
    image_type = ask_image_type()
    order_mode = ask_order_mode()
    prefix = f"gray_leaf_spot_{source_abbr}_{image_type}_"

    rename_images(SOURCE_DIR, prefix, order_mode)
