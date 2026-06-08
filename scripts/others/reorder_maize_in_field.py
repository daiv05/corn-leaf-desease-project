#!/usr/bin/env python3
import argparse
import csv
import shutil
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Move maize-in-field images into disease folders based on the CSV labels."
    )
    parser.add_argument(
        "--csv",
        default="data/raw/maize-in-field-dataset/Kaggle Dataset/Database.csv",
        help="Path to Database.csv",
    )
    parser.add_argument(
        "--images",
        default="data/raw/maize-in-field-dataset/Kaggle Dataset/leaf_images",
        help="Path to the leaf_images directory",
    )
    parser.add_argument(
        "--output",
        default="data/clean/maize-in-field-dataset",
        help="Output directory for organized images",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without moving files",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    csv_path = Path(args.csv)
    images_dir = Path(args.images)
    output_dir = Path(args.output)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    moved = 0
    missing_files = 0
    multi_label = 0
    missing_label = 0
    counts = {}

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        label_columns = [
            col
            for col in reader.fieldnames
            if col not in {"imgID_id", "filePath"}
        ]

        for row in reader:
            file_name = row.get("filePath", "").strip()
            if not file_name:
                missing_label += 1
                continue

            labels = [
                col for col in label_columns if row.get(col, "0").strip() == "1"
            ]

            if len(labels) == 0:
                target_folder = "missing_label"
                missing_label += 1
            elif len(labels) > 1:
                target_folder = "multi_label"
                multi_label += 1
            else:
                target_folder = labels[0]

            src = images_dir / file_name
            if not src.exists():
                missing_files += 1
                continue

            dest_dir = output_dir / target_folder
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name

            if args.dry_run:
                print(f"DRY RUN: {src} -> {dest}")
            else:
                shutil.move(str(src), str(dest))

            counts[target_folder] = counts.get(target_folder, 0) + 1
            moved += 1

    print("Done")
    print(f"Moved: {moved}")
    print(f"Missing files: {missing_files}")
    print(f"Multi-label: {multi_label}")
    print(f"Missing label: {missing_label}")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")

if __name__ == "__main__":
    main()
