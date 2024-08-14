import argparse
import json
import logging
from pathlib import Path

from torchvision.datasets import FashionMNIST

from image_classifier.logger.logger import initialize_root_logger

initialize_root_logger()
logger = logging.getLogger("PREPARE DATA")
logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, help="Path to the output data folder.", required=True)

    known_args, unknown_args = parser.parse_known_args()

    known_args_with_type = {
        arg: {"value": str(value), "type": f"{type(value)}"} for arg, value in known_args.__dict__.items()
    }
    logger.info(msg=f"PARSED KNOWN ARGUMENTS:\n{json.dumps(obj=known_args_with_type, indent=4)}\n")

    if unknown_args:
        # For debugging purposes. If any unknown args occur, it must be a user mistake.
        logger.warning(msg=f"PARSED UNKNOWN ARGUMENTS:\n{' '.join(unknown_args)}\n")

    return known_args


def download_data(data_dir: Path) -> None:
    logger.info(f"Downloading FashionMNIST data to {data_dir}.")
    FashionMNIST(root=data_dir, train=True, download=True)
    FashionMNIST(root=data_dir, train=False, download=True)


def clean_and_transform_data(data_dir: Path) -> None:
    logger.info(f"Cleaning and transforming FashionMNIST data in {data_dir}.")


def main() -> None:
    args = parse_args()

    download_data(data_dir=args.data_dir)
    clean_and_transform_data(data_dir=args.data_dir)


if __name__ == "__main__":
    main()
