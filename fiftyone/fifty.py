import time
import fiftyone as fo

fo.config.do_not_track = True
fo.config.default_app_port = 5151
fo.config.max_thread_pool_workers = 2

datasets = fo.list_datasets()

print("Available datasets:")
for d in datasets:
    print("-", d)

if not datasets:
    raise ValueError("No datasets found")

# Abrir primero disponible
dataset = fo.load_dataset(datasets[0])

session = fo.launch_app(
    dataset=dataset,
    address="0.0.0.0",
    port=5151,
    auto=False,
)

print(session.url)

while True:
    time.sleep(60)

