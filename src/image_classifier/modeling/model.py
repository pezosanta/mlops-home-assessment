import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Setting the logging level of other loggers to WARNING
logging.basicConfig(
    format="%(asctime)s - %(name)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s",
    encoding="utf-8",
    level=logging.WARNING,
    datefmt="%Y-%m-%d %H:%M:%S [%Z]",
)
logger = logging.getLogger("MODEL")
logger.setLevel(logging.INFO)


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_model(
    pretrained_model_path: str | None = None, base_path: Path = Path(__file__).parents[3].joinpath("experiments")
) -> nn.Module:

    # Handling model structure based hyperparameters here (if there were any)
    logger.info(msg="Composing and returning model.")
    model = Net()

    if pretrained_model_path:
        full_model_path = base_path.joinpath(pretrained_model_path)

        logger.info(msg=f"Loading pretrained model ({str(full_model_path)}).")
        model.load_state_dict(state_dict=torch.load(full_model_path))

    return model


def save_model(
    model: Net, experiment_name: str, run_name: str, base_path: Path = Path(__file__).parents[3].joinpath("experiments")
) -> None:

    run_path = base_path.joinpath(f"{experiment_name}/{run_name}")
    run_path.mkdir(parents=True, exist_ok=True)

    logger.info(msg=f"Saving model to {str(run_path)}.")
    torch.save(model.state_dict(), run_path.joinpath("model.pth"))
