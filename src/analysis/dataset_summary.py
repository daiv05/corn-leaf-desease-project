"""
Resumen del dataset: conteo por clase/entorno y tamaño en disco.

Funciones públicas:
  count_clean_dataset(clean_dir)  - DataFrame con columnas Clase, Lab, Real, Total
  print_disk_summary(dataset_root) - reporte de disco por subcarpeta raíz
"""

from pathlib import Path

import pandas as pd

from src.config import DATASET_ROOT

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_COL_W = 52


# ---------------------------------------------------------------------------
# Funciones de conteo
# ---------------------------------------------------------------------------


def count_clean_dataset(clean_dir: Path) -> pd.DataFrame:
    """
    Cuenta imágenes en clean/<clase>/{lab,real}/.

    Returns:
        DataFrame con columnas Clase, Lab, Real, Total - una fila por clase.
    """
    if not clean_dir.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {clean_dir}")

    rows = []
    for class_dir in sorted(clean_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        lab_count = len(list((class_dir / "lab").iterdir())) if (class_dir / "lab").exists() else 0
        real_count = (
            len(list((class_dir / "real").iterdir())) if (class_dir / "real").exists() else 0
        )
        rows.append(
            {
                "Clase": class_dir.name,
                "Lab": lab_count,
                "Real": real_count,
                "Total": lab_count + real_count,
            }
        )

    return pd.DataFrame(rows)


def _df_to_markdown(df: pd.DataFrame) -> str:
    col_widths = {col: max(len(col), df[col].astype(str).str.len().max()) for col in df.columns}
    header = " | ".join(f"{col:<{col_widths[col]}}" for col in df.columns)
    separator = " | ".join("-" * col_widths[col] for col in df.columns)
    rows = [
        " | ".join(f"{str(val):<{col_widths[col]}}" for col, val in zip(df.columns, row))
        for row in df.itertuples(index=False)
    ]
    return "\n".join([header, separator] + rows)


def print_class_table(clean_dir: Path) -> None:
    """Imprime tabla Markdown con conteo por clase y entorno."""
    df = count_clean_dataset(clean_dir)

    print("\n### Conteo por clase\n")
    print(_df_to_markdown(df))
    print("\n### Resumen global")
    print(f"* **Total Lab:**   {df['Lab'].sum()}")
    print(f"* **Total Real:**  {df['Real'].sum()}")
    print(f"* **Total:**       {df['Total'].sum()}")


# ---------------------------------------------------------------------------
# Funciones de disco
# ---------------------------------------------------------------------------


def _fmt_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def _scan(directory: Path) -> tuple[int, int]:
    count = size = 0
    for f in directory.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            count += 1
            size += f.stat().st_size
    return count, size


def _row(label: str, count: int, size: int, indent: int = 0) -> str:
    prefix = "  " * indent
    return f"  {prefix + label:<{_COL_W}}  {count:>6}  {_fmt_size(size):>10}"


def _sep(char: str = "─", width: int = 76) -> str:
    return "  " + char * width


def _print_section(title: str, directory: Path, depth: int = 1) -> tuple[int, int]:
    if not directory.exists():
        print(f"\n  [!] No encontrado: {directory}")
        return 0, 0

    subdirs = sorted(d for d in directory.iterdir() if d.is_dir())
    total_c, total_s = _scan(directory)

    print(f"\n  {title}")
    print(_sep())
    print(f"  {'Carpeta':<{_COL_W}}  {'Imgs':>6}  {'Tamaño':>10}")
    print(_sep())

    for sub in subdirs:
        c, s = _scan(sub)
        print(_row(sub.name, c, s, indent=0))
        if depth >= 2:
            for n in sorted(d for d in sub.iterdir() if d.is_dir()):
                nc, ns = _scan(n)
                print(_row(n.name, nc, ns, indent=1))

    print(_sep())
    print(_row("TOTAL", total_c, total_s))
    return total_c, total_s


def print_disk_summary(dataset_root: Path) -> None:
    """Imprime reporte de disco por subcarpeta raíz (clean/, raw/, etc.)."""
    if not dataset_root.exists():
        raise FileNotFoundError(f"DATASET_ROOT no encontrado: {dataset_root}")

    print()
    print("=" * 78)
    print(f"  Dataset: {dataset_root}")
    print("=" * 78)

    grand_c = grand_s = 0
    for top in sorted(d for d in dataset_root.iterdir() if d.is_dir()):
        if top.name == "clean":
            c, s = _print_section("clean/  (por enfermedad - lab / real)", top, depth=2)
        elif top.name == "raw":
            c, s = _print_section("raw/  (por fuente)", top, depth=1)
        else:
            c, s = _scan(top)
            print(f"\n  {top.name}/")
            print(_sep())
            print(_row(top.name, c, s))
            print(_sep())
            print(_row("TOTAL", c, s))
        grand_c += c
        grand_s += s

    print()
    print("=" * 78)
    print(_row("TOTAL GLOBAL", grand_c, grand_s))
    print("=" * 78)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Resumen del dataset de maíz.")
    parser.add_argument(
        "--mode",
        choices=["table", "disk", "all"],
        default="all",
        help="table: conteo por clase | disk: uso en disco | all: ambos (default)",
    )
    args = parser.parse_args()

    clean_dir = DATASET_ROOT / "clean"

    if args.mode in ("table", "all"):
        print_class_table(clean_dir)
    if args.mode in ("disk", "all"):
        print_disk_summary(DATASET_ROOT)


if __name__ == "__main__":
    main()
