import argparse
import time

import fiftyone as fo

DEFAULT_PORT = 5151
DEFAULT_LIMIT = 500

fo.config.do_not_track = True
fo.config.default_app_port = DEFAULT_PORT
fo.config.max_thread_pool_workers = 2

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch FiftyOne app")
    parser.add_argument("--dataset", default=None, help="Dataset name to load")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port for the app")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Max samples to load in the view (0 = no limit)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = fo.list_datasets()

    print("Available datasets:")
    for d in datasets:
        print("-", d)

    if not datasets:
        raise ValueError("No datasets found")

    dataset_name = args.dataset or datasets[0]
    dataset = fo.load_dataset(dataset_name)

    view = dataset
    if args.limit and args.limit > 0:
        view = dataset.limit(args.limit)

    session = fo.launch_app(
        dataset=view,
        address="0.0.0.0",
        port=args.port,
        auto=False,
    )

    print(session.url)

    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()

