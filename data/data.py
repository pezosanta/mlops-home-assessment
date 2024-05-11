from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

def get_train_test_data_loader(
    db_host: str,
    db_name: str,
    db_user: str,
    db_pass: str,
    batch_size: int = 4,
    num_workers: int = 2,
    training_data_path: Path = Path(__file__).parent.joinpath("cifar_dataset")
) -> DataLoader:
    """
    For the purpose of this exercice, this function is here to simulate a call to a database.
    You can keep it like that but keep in mind in a real situation, here you would have the code for db communication.
    """
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=training_data_path,
        train=True,
        download=True,
        transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    testset = torchvision.datasets.CIFAR10(
        root=training_data_path,
        train=False,
        download=True,
        transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return trainloader, testloader