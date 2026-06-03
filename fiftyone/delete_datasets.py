import fiftyone as fo

datasets = fo.list_datasets()

if not datasets:
    print("No datasets found")
    exit()

print("Datasets to delete:")

for d in datasets:
    print(f" - {d}")

confirm = input("\nDelete ALL datasets? (yes/no): ")

if confirm.lower() != "yes":
    print("Cancelled")
    exit()

for d in datasets:
    try:
        fo.delete_dataset(d)
        print(f"Deleted: {d}")

    except Exception as e:
        print(f"Error deleting {d}: {e}")

print("\nAll datasets removed")

