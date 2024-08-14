import argparse
import json
import logging
from pathlib import Path

from image_classifier.data.prepare_data import download_data
from image_classifier.logger.logger import initialize_root_logger
from image_classifier.modeling.evaluate import evaluate
from image_classifier.modeling.train import train

initialize_root_logger()
logger = logging.getLogger("TRAINING")
logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, help="Path to the input data folder.", required=True)
    parser.add_argument("--checkpoint_dir", type=Path, help="Path to the output checkpoint folder.", required=True)
    parser.add_argument(
        "--horizontal_flip_prob", type=float, help="Probability of RandomHorizontalFlip augmentation.", required=True
    )
    parser.add_argument(
        "--rotation_degrees", type=int, help="Rotation degree of RandomRotation augmentation.", required=True
    )
    parser.add_argument("--learning_rate", type=float, help="Learning Rate hyperparameter.", required=True)
    parser.add_argument("--batch_size", type=int, help="Batch size to load the data with.")
    parser.add_argument(
        "--num_workers", type=int, help="Number of workers (CPU threads/processes) to load the data with."
    )
    parser.add_argument("--prefetch_factor", type=int, help="Number of batches loaded in advance by each worker.")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs to train the model with.")
    parser.add_argument("--dev_mode", type=bool, help="Run the training in development mode.")

    known_args, unknown_args = parser.parse_known_args()

    known_args_with_type = {
        arg: {"value": str(value), "type": f"{type(value)}"} for arg, value in known_args.__dict__.items()
    }
    logger.info(msg=f"PARSED KNOWN ARGUMENTS:\n{json.dumps(obj=known_args_with_type, indent=4)}\n")

    if unknown_args:
        # For debugging purposes. If any unknown args occur, it must be a user mistake.
        logger.warning(msg=f"PARSED UNKNOWN ARGUMENTS:\n{' '.join(unknown_args)}\n")

    return known_args


def main() -> None:
    args = parse_args()

    download_data(data_dir=args.data_dir)

    train(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        horizontal_flip_prob=args.horizontal_flip_prob,
        rotation_degrees=args.rotation_degrees,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        num_epochs=args.num_epochs,
        dev_mode=args.dev_mode,
    )

    evaluate(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        dev_mode=args.dev_mode,
    )


if __name__ == "__main__":
    main()
