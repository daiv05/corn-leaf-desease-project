"""
Reorganiza las imágenes de maize-in-field-dataset en subcarpetas por enfermedad
según Database.csv. Imágenes con múltiples etiquetas van a multi_label/.

Fuente:  .../raw/maize-in-field-dataset/Kaggle Dataset/leaf_images/
Destino: .../raw/maize-in-field-dataset/Kaggle Dataset/  (subcarpetas por etiqueta)

Ejecutar desde cualquier directorio:
    python scripts/cleanup/others/sort_maize_by_disease.py
    python scripts/cleanup/others/sort_maize_by_disease.py --dry-run   # solo muestra lo que haría
"""

import argparse
import csv
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

KAGGLE_DIR = Path("/mnt/datasets/data/corn-leaf-diseases/raw/maize-in-field-dataset/Kaggle Dataset")
CSV_FILE = KAGGLE_DIR / "Database.csv"
SRC_DIR = KAGGLE_DIR / "leaf_images"
DST_ROOT = KAGGLE_DIR  # subcarpetas creadas aquí

LABEL_COLS = ["GLS", "NCLB", "PLS", "CR", "SR", "NoFoliarSymptoms", "Other", "UnidentifiedDisease"]
MULTI_DIR = "multi_label"


def build_move_plan(csv_path: Path) -> list[tuple[Path, Path]]:
    """Lee el CSV y devuelve lista de (src, dst) para cada imagen."""
    plan = []
    missing = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filePath"].strip()
            src = SRC_DIR / filename

            active = [col for col in LABEL_COLS if row[col].strip() == "1"]

            if len(active) == 0:
                folder = "no_label"
            elif len(active) == 1:
                folder = active[0]
            else:
                folder = MULTI_DIR

            dst = DST_ROOT / folder / filename

            if not src.exists():
                missing.append(str(src))
                continue

            plan.append((src, dst))

    if missing:
        print(f"ADVERTENCIA: {len(missing)} archivo(s) en el CSV no encontrados en disco:")
        for m in missing[:10]:
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... y {len(missing) - 10} más")

    return plan


def check_collisions(plan: list[tuple[Path, Path]]) -> bool:
    """Detecta si dos fuentes distintas irían al mismo destino."""
    dst_counter: dict[Path, list[Path]] = defaultdict(list)
    for src, dst in plan:
        dst_counter[dst].append(src)

    collisions = {dst: srcs for dst, srcs in dst_counter.items() if len(srcs) > 1}
    if collisions:
        print(f"ERROR: {len(collisions)} colisión(es) de destino detectadas:")
        for dst, srcs in list(collisions.items())[:5]:
            print(f"  -> {dst}")
            for s in srcs:
                print(f"       desde: {s}")
        return False
    return True


def execute_plan(plan: list[tuple[Path, Path]], dry_run: bool) -> None:
    folders_created: set[Path] = set()
    moved = 0
    skipped = 0

    for src, dst in plan:
        if dst.exists():
            # Ya está en su lugar (idempotente)
            skipped += 1
            continue

        if not dry_run:
            if dst.parent not in folders_created:
                dst.parent.mkdir(parents=True, exist_ok=True)
                folders_created.add(dst.parent)
            shutil.move(str(src), str(dst))

        moved += 1

    action = "Moverían" if dry_run else "Movidos"
    print(f"{action}: {moved}  |  Ya en destino (omitidos): {skipped}")


def print_summary(plan: list[tuple[Path, Path]]) -> None:
    folder_counts: Counter = Counter()
    for _, dst in plan:
        folder_counts[dst.parent.name] += 1

    print("\nDistribución por carpeta destino:")
    for folder, count in sorted(folder_counts.items()):
        print(f"  {folder:<22} {count:>5} imágenes")
    print(f"  {'TOTAL':<22} {sum(folder_counts.values()):>5} imágenes")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reorganiza maize-in-field-dataset por enfermedad."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Muestra qué se haría sin mover nada."
    )
    args = parser.parse_args()

    for path, label in [(CSV_FILE, "CSV"), (SRC_DIR, "directorio de imágenes")]:
        if not path.exists():
            sys.exit(f"ERROR: {label} no encontrado: {path}")

    print(f"CSV:     {CSV_FILE}")
    print(f"Fuente:  {SRC_DIR}")
    print(f"Destino: {DST_ROOT}/<enfermedad>/")
    if args.dry_run:
        print("MODO DRY-RUN - no se moverá nada\n")
    else:
        print()

    plan = build_move_plan(CSV_FILE)
    print(f"Entradas válidas en el plan: {len(plan)}")

    if not check_collisions(plan):
        sys.exit(1)

    print_summary(plan)
    print()

    execute_plan(plan, dry_run=args.dry_run)

    if not args.dry_run:
        remaining = list(SRC_DIR.iterdir())
        if not remaining:
            print(f"\nDirectorio fuente vacío tras el movimiento: {SRC_DIR}")
        else:
            print(f"\nArchivos que quedaron en {SRC_DIR.name}/: {len(remaining)}")
            for f in remaining[:5]:
                print(f"  {f.name}")


if __name__ == "__main__":
    main()
