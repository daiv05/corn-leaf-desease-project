"""
Mueve las imágenes listadas en tmp/to_move.md desde data/raw/ hacia
data/clean/common_rust/lab/multicrop/, conservando el nombre original.
"""

import os
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TO_MOVE_MD = os.path.join(PROJECT_ROOT, "tmp", "to_move.md")
DEST_DIR = os.path.join(PROJECT_ROOT, "data", "clean", "common_rust", "lab", "multicrop")


def main() -> None:
    os.makedirs(DEST_DIR, exist_ok=True)

    with open(TO_MOVE_MD, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    moved = 0
    skipped = 0
    missing = 0

    for rel_path in lines:
        src = os.path.join(PROJECT_ROOT, rel_path)
        filename = os.path.basename(rel_path)
        dst = os.path.join(DEST_DIR, filename)

        if not os.path.isfile(src):
            print(f"[NO ENCONTRADO] {rel_path}")
            missing += 1
            continue

        if os.path.exists(dst):
            print(f"[YA EXISTE]    {filename}")
            skipped += 1
            continue

        shutil.copy2(src, dst)
        moved += 1

    print(f"\nMovidas:        {moved}")
    print(f"Ya existían:    {skipped}")
    print(f"No encontradas: {missing}")
    print(f"Destino:        {DEST_DIR}")


if __name__ == "__main__":
    main()
