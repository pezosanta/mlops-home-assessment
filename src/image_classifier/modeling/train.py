import argparse
import json
import logging
import os
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from image_classifier.data.data import FashionMNISTDataModule
from image_classifier.logger.logger import initialize_root_logger
from image_classifier.modeling.model import SimpleNet
from image_classifier.modeling.orchestrator import Orchestrator

initialize_root_logger()
logger = logging.getLogger("TRAINING")
logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    def str2bool(v: str) -> bool:
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

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
    parser.add_argument("--dev_mode", type=str2bool, help="Run the training in development mode.")

    known_args, unknown_args = parser.parse_known_args()

    known_args_with_type = {
        arg: {"value": str(value), "type": f"{type(value)}"} for arg, value in known_args.__dict__.items()
    }
    logger.info(msg=f"PARSED KNOWN ARGUMENTS:\n{json.dumps(obj=known_args_with_type, indent=4)}\n")

    if unknown_args:
        # For debugging purposes. If any unknown args occur, it must be a user mistake.
        logger.warning(msg=f"PARSED UNKNOWN ARGUMENTS:\n{' '.join(unknown_args)}\n")

    return known_args


def train(
    data_dir: Path,
    checkpoint_dir: Path,
    horizontal_flip_prob: float,
    rotation_degrees: int,
    learning_rate: float,
    batch_size: int = 128,
    num_workers: int = 4,
    prefetch_factor: int = 4,
    num_epochs: int = 10,
    dev_mode: bool = False,
) -> None:
    L.seed_everything(seed=42)

    logger.info(msg="Initializing MLFlowLogger.")
    mlflow_logger = MLFlowLogger(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME"),
        run_id=os.getenv("MLFLOW_RUN_ID"),
    )

    logger.info(msg="Initializing Data Module.")
    dm = FashionMNISTDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        horizontal_flip_prob=horizontal_flip_prob,
        rotation_degrees=rotation_degrees,
    )

    logger.info(msg="Initializing Orchestrator.")
    model = SimpleNet()
    lightning_module = Orchestrator(model=model, classes=dm.CLASSES, learning_rate=learning_rate)

    logger.info(msg="Initializing Model Checkpoint.")
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir, filename="{epoch}-{val_loss:.2f}-{val_f1_score:.2f}", save_top_k=1
    )

    logger.info(msg="Initializing Trainer.")
    trainer = L.Trainer(
        deterministic=True,
        max_epochs=num_epochs,
        accelerator="auto",
        precision="16-mixed",
        callbacks=[checkpoint_callback],
        logger=mlflow_logger,
        limit_train_batches=10 if dev_mode else None,
        limit_val_batches=10 if dev_mode else None,
    )

    logger.info(msg="Starting training.")
    trainer.fit(lightning_module, dm)


def main() -> None:
    args = parse_args()

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


if __name__ == "__main__":
    main()
