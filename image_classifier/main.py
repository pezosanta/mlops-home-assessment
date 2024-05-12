import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from image_classifier.data.data import get_db_connect_object, get_train_test_data_loader
from image_classifier.model.model import get_model, save_model
from image_classifier.model.train import train
from image_classifier.model.validation import validation
from image_classifier.visualization.plot import save_metrics_plot

# Setting the logging level of other loggers to WARNING
logging.basicConfig(
    format="%(asctime)s - %(name)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s",
    encoding="utf-8",
    level=logging.WARNING,
    datefmt="%Y-%m-%d %H:%M:%S [%Z]",
)
logger = logging.getLogger("MAIN")
logger.setLevel(logging.INFO)


def check_if_run_name_exists(
    experiment_name: str, run_name: str, base_path: Path = Path(__file__).parents[1].joinpath("experiments")
) -> None:

    run_path = base_path.joinpath(f"{experiment_name}/{run_name}")

    if run_path.exists():
        logger.error(msg=f"RUN NAME {str(run_path)} ALREADY EXISTS. ABORTING.")
        raise ValueError


def orchestrator(
    experiment_name: str,
    run_name: str,
    learning_rate: float,
    momentum: float,
    num_epochs: int,
    num_workers: int,
    batch_size: int,
    db_pass: str,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
) -> None:

    logger.info(f"Using device: {torch.device(device=device).type}")

    trainloader, testloader, classes = get_train_test_data_loader(
        db_connect=get_db_connect_object(db_pass=db_pass), batch_size=batch_size, num_workers=num_workers
    )

    model = get_model()
    model.to(device=torch.device(device=device))

    logger.info(msg="Initializing Cross-Entropy Loss and SGD optimizer.")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    train(
        trainloader=trainloader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
    )

    classwise_accuracy = validation(testloader=testloader, model=model, classes=classes, device=device)

    save_model(model=model, experiment_name=experiment_name, run_name=run_name)
    save_metrics_plot(metrics=classwise_accuracy, experiment_name=experiment_name, run_name=run_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name", type=str, help="Experiment name to save the model and validation metrics to."
    )
    parser.add_argument(
        "--run_name", type=str, help="Run name to save the model and validation metrics to. MUST BE UNIQUE!"
    )
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning Rate for the SGD optimizer.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for the SGD optimizer.")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs to train the model for.")
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Number of workers (CPU threads/processes) to load the data with."
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Epoch size to load the data with.")
    parser.add_argument("--db_pass", type=str, help="DB password for connection.")

    known_args, unknown_args = parser.parse_known_args()

    known_args_with_type = {
        arg: {"value": str(value), "type": f"{type(value)}"} for arg, value in known_args.__dict__.items()
    }
    logger.info(msg=f"PARSED KNOWN ARGUMENTS:\n{json.dumps(obj=known_args_with_type, indent=4)}\n")

    if unknown_args:
        # For debugging purposses. If any unknown args occur, it must be a user mistake.
        logger.warning(msg=f"PARSED UNKNOWN ARGUMENTS:\n{' '.join(unknown_args)}\n")

    return known_args


def main() -> None:
    args = parse_args()

    check_if_run_name_exists(experiment_name=args.experiment_name, run_name=args.run_name)

    orchestrator(
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        db_pass=args.db_pass,
    )


if __name__ == "__main__":
    main()
