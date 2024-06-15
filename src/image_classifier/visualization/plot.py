import logging
from pathlib import Path

import matplotlib.pyplot as plt

# Setting the logging level of other loggers to WARNING
logging.basicConfig(
    format="%(asctime)s - %(name)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s",
    encoding="utf-8",
    level=logging.WARNING,
    datefmt="%Y-%m-%d %H:%M:%S [%Z]",
)
logger = logging.getLogger("PLOT")
logger.setLevel(logging.INFO)


def add_bar_values(classes: list[str], values: list[float]) -> None:
    for i in range(len(classes)):
        plt.text(i, values[i] + 1, f"{values[i]:.2f}", ha="center")


def save_metrics_plot(
    metrics: dict[str, float],
    experiment_name: str,
    run_name: str,
    base_path: Path = Path(__file__).parents[3].joinpath("experiments"),
) -> None:

    run_path = base_path.joinpath(f"{experiment_name}/{run_name}")
    run_path.mkdir(parents=True, exist_ok=True)

    logger.info(msg=f"Saving metrics plot to {str(run_path)}.")

    classes = list(metrics.keys())
    values = list(metrics.values())

    _ = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(x=classes, height=values, color="maroon", width=0.4)

    add_bar_values(classes=classes, values=values)

    plt.xlabel("Classes")
    plt.ylabel("Validation Accuracy [%]")
    plt.title("Class-wise and Total Validation Accuracy")
    plt.savefig(run_path.joinpath("metrics.png"), bbox_inches="tight")
