import os
import fiftyone as fo

BASE_DIR = "/mnt/volume-nbg1-1/data/corn-leaf-diseases/raw"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
UNLABELED_DIR = "_unlabeled"


def iter_classification_samples(dataset_dir):
    samples = []

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

            samples.append(sample)

    return samples

for folder in os.listdir(BASE_DIR):

    path = os.path.join(BASE_DIR, folder)

    if not os.path.isdir(path):
        continue

    dataset_name = folder.lower().replace(" ", "_")

    if fo.dataset_exists(dataset_name):
        print(f"Skipping existing dataset: {dataset_name}")
        continue

    print(f"Importing: {dataset_name}")

    samples = iter_classification_samples(path)
    if not samples:
        print(f"No image samples found, skipping: {dataset_name}")
        continue

    dataset = fo.Dataset(name=dataset_name)
    dataset.add_samples(samples)

    dataset.persistent = True

    print(f"Imported: {dataset_name}")


