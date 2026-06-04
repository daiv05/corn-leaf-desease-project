import argparse
import os
from typing import Iterator, List

import fiftyone as fo

BASE_DIR = "/mnt/datasets/data/corn-leaf-diseases/raw"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
UNLABELED_DIR = "_unlabeled"
DEFAULT_BATCH_SIZE = 64


def iter_classification_samples(dataset_dir: str) -> Iterator[fo.Sample]:
    for root, _, files in os.walk(dataset_dir):
        for filename in files:
            _, ext = os.path.splitext(filename)
            if ext.lower() not in IMAGE_EXTS:
                continue

            filepath = os.path.join(root, filename)
            relpath = os.path.relpath(filepath, dataset_dir)
            label = relpath.split(os.path.sep, 1)[0]

            if label.startswith("."):
                continue

            if label == UNLABELED_DIR:
                label = None

            sample = fo.Sample(filepath=filepath)
            if label is not None:
                sample["ground_truth"] = fo.Classification(label=label)

            yield sample


def iter_batches(samples: Iterator[fo.Sample], batch_size: int) -> Iterator[List[fo.Sample]]:
    batch: List[fo.Sample] = []
    for sample in samples:
        batch.append(sample)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import datasets into FiftyOne")
    parser.add_argument("--base-dir", default=BASE_DIR, help="Root directory for datasets")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Samples per batch")
    parser.add_argument("--skip-existing", action="store_true", help="Skip datasets that already exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    batch_size = max(1, args.batch_size)

    for folder in os.listdir(base_dir):

        path = os.path.join(base_dir, folder)

        if not os.path.isdir(path):
            continue

        dataset_name = folder.lower().replace(" ", "_")

        if fo.dataset_exists(dataset_name):
            if args.skip_existing:
                print(f"Skipping existing dataset: {dataset_name}")
                continue
            fo.delete_dataset(dataset_name)

        print(f"Importing: {dataset_name}")

        dataset = fo.Dataset(name=dataset_name)
        added = 0
        for batch in iter_batches(iter_classification_samples(path), batch_size):
            dataset.add_samples(batch)
            added += len(batch)

        if added == 0:
            print(f"No image samples found, deleting empty dataset: {dataset_name}")
            fo.delete_dataset(dataset_name)
            continue

        dataset.persistent = True
        print(f"Imported: {dataset_name} ({added} samples)")


if __name__ == "__main__":
    main()


