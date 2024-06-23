import logging
from pathlib import Path

import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from image_classifier.utils.file_loaders import load_json

logger = logging.getLogger("DATA")
logger.setLevel(logging.DEBUG)


def get_db_connect_object(db_pass: str, db_config: Path = Path(__file__).parent.joinpath("db_connect.yaml")) -> object:
    """
    Mocking a db connection.
    In a real world scenario, db password would be acquired from a Secret Manager / Key Vault service.
    """

    def mocked_db_connector(db_host: str, db_name: str, db_user: str, db_pass: str) -> object:
        return object

    logger.debug(msg="This is a debug message.")
    logger.info(msg="This is a info message.")
    logger.warning(msg="This is a warning message.")
    logger.error(msg="This is a error message.")
    logger.critical(msg="This is a critical message.")
    logger.info(msg="Getting the DB connetion object.")

    # Getting non-secret db infos from config file
    with open(file=db_config, mode="r") as f:
        db_config_dict = yaml.safe_load(f)
        db_host = db_config_dict["db"]["host"]
        db_name = db_config_dict["db"]["name"]
        db_user = db_config_dict["db"]["user"]

    # creating mocked db connector object
    db_connector = mocked_db_connector(db_host=db_host, db_name=db_name, db_user=db_user, db_pass=db_pass)

    return db_connector


def get_train_test_data_loader(
    db_connect: object,
    batch_size: int = 4,
    num_workers: int = 2,
    local_dataset_path: Path = Path(__file__).parents[3].joinpath("data/cifar_dataset"),
) -> tuple[DataLoader, DataLoader, list[str]]:
    """
    For the purpose of this exercice, this function is here to simulate a call to a database.
    You can keep it like that but keep in mind in a real situation, here you would have the code for db communication.
    """
    _ = db_connect

    logger.info(f"Loading Dataset from {local_dataset_path}.")

    def get_classes() -> list[str]:
        logger.info(msg="Getting dataset classes.")

        return load_json(file_path=local_dataset_path.joinpath("metadata.json"))["classes"]

    logger.info(msg=f"Creating Train DataLoader with batch_size={batch_size}, num_workers={num_workers}.")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10(root=local_dataset_path, train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    logger.info(msg=f"Creating Test DataLoader with batch_size={batch_size}, num_workers={num_workers}.")
    testset = CIFAR10(root=local_dataset_path, train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader, get_classes()
