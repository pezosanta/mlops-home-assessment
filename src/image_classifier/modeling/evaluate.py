import argparse
import json
import logging
from pathlib import Path

import lightning as L
from lightning.pytorch.loggers import MLFlowLogger

from image_classifier.data.data import FashionMNISTDataModule
from image_classifier.logger.logger import initialize_root_logger
from image_classifier.modeling.model import SimpleNet
from image_classifier.modeling.orchestrator import Orchestrator

initialize_root_logger()
logger = logging.getLogger("EVALUATION")
logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, help="Path to the input data folder.", required=True)
    parser.add_argument("--checkpoint_dir", type=Path, help="Path to the input checkpoint folder.", required=True)
    parser.add_argument("--batch_size", type=int, help="Batch size to load the data with.")
    parser.add_argument(
        "--num_workers", type=int, help="Number of workers (CPU threads/processes) to load the data with."
    )
    parser.add_argument("--prefetch_factor", type=int, help="Number of batches loaded in advance by each worker.")
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


def get_best_checkpoint(ckpt_dir: Path) -> Path:
    # epoch=1-val_loss=0.00-val_f1_score=0.00
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    ckpts.sort(key=lambda x: float(x.stem.rsplit(sep="=", maxsplit=1)[1]), reverse=True)
    return ckpts[0]


def evaluate(
    data_dir: Path,
    checkpoint_dir: Path,
    batch_size: int = 128,
    num_workers: int = 4,
    prefetch_factor: int = 4,
    dev_mode: bool = False,
) -> None:
    L.seed_everything(seed=42)

    logger.info(msg="Initializing MLFlowLogger.")
    mlflow_logger = MLFlowLogger(experiment_name="fashion_mnist_test_1", run_name="test_log_model")

    logger.info(msg="Initializing Data Module.")
    dm = FashionMNISTDataModule(
        data_dir=data_dir,
        local_experiment=True,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    logger.info(msg="Initializing Orchestrator.")
    model = SimpleNet()
    lightning_module = Orchestrator(model=model, classes=dm.CLASSES)

    logger.info(msg="Initializing Trainer.")
    trainer = L.Trainer(
        deterministic=True,
        accelerator="auto",
        precision="16-mixed",
        logger=mlflow_logger,
        limit_test_batches=2 if dev_mode else None,
    )

    best_checkpoint = get_best_checkpoint(ckpt_dir=checkpoint_dir)
    logger.info(msg=f"Starting evaluation with {best_checkpoint}.")
    trainer.test(lightning_module, dm, ckpt_path=best_checkpoint)


def main() -> None:
    args = parse_args()

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
