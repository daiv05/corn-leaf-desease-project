"""
Resumen del dataset: cantidad de imágenes y tamaño en disco.

Muestra:
  - Totales globales por subcarpeta raíz (clean/, raw/, etc.)
  - Para clean/: desglose por enfermedad y luego por lab/ o real/
  - Para raw/:   desglose por fuente de dataset
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import DATASET_ROOT

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

COL_W = 52  # ancho de columna de nombre


def fmt_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def scan(directory: Path) -> tuple[int, int]:
    """Devuelve (cantidad_imágenes, bytes_totales) bajo directory."""
    count = size = 0
    for f in directory.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            count += 1
            size += f.stat().st_size
    return count, size


def row(label: str, count: int, size: int, indent: int = 0) -> str:
    prefix = "  " * indent
    label_col = f"{prefix}{label}"
    return f"  {label_col:<{COL_W}}  {count:>6}  {fmt_size(size):>10}"


def separator(char: str = "─", width: int = 76) -> str:
    return "  " + char * width


def print_section(title: str, directory: Path, depth: int = 1) -> tuple[int, int]:
    """
    Imprime una sección con desglose de subdirectorios hasta `depth` niveles.
    Retorna (total_count, total_size) de la sección.
    """
    if not directory.exists():
        print(f"\n  [!] No encontrado: {directory}")
        return 0, 0

    subdirs = sorted([d for d in directory.iterdir() if d.is_dir()])
    total_c, total_s = scan(directory)

    print(f"\n  {title}")
    print(separator())
    print(f"  {'Carpeta':<{COL_W}}  {'Imgs':>6}  {'Tamaño':>10}")
    print(separator())

    for sub in subdirs:
        c, s = scan(sub)
        print(row(sub.name, c, s, indent=0))

        if depth >= 2:
            nested = sorted([d for d in sub.iterdir() if d.is_dir()])
            for n in nested:
                nc, ns = scan(n)
                print(row(n.name, nc, ns, indent=1))

    print(separator())
    print(row("TOTAL", total_c, total_s, indent=0))
    return total_c, total_s


def main() -> None:
    if not DATASET_ROOT.exists():
        raise SystemExit(f"DATASET_ROOT no encontrado: {DATASET_ROOT}")

    print()
    print("=" * 78)
    print(f"  Dataset: {DATASET_ROOT}")
    print("=" * 78)

    grand_total_c = grand_total_s = 0

    top_dirs = sorted([d for d in DATASET_ROOT.iterdir() if d.is_dir()])
    for top in top_dirs:
        name = top.name
        if name == "clean":
            # clean/ → enfermedad → lab/ | real/
            c, s = print_section("clean/  (por enfermedad → lab / real)", top, depth=2)
        elif name == "raw":
            # raw/ → fuente de dataset (sin entrar más)
            c, s = print_section("raw/  (por fuente)", top, depth=1)
        else:
            c, s = scan(top)
            print(f"\n  {name}/")
            print(separator())
            print(row(name, c, s))
            print(separator())
            print(row("TOTAL", c, s))

        grand_total_c += c
        grand_total_s += s

    print()
    print("=" * 78)
    print(row("TOTAL GLOBAL", grand_total_c, grand_total_s))
    print("=" * 78)
    print()


if __name__ == "__main__":
    main()
