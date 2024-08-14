import logging
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST

logger = logging.getLogger("DATA")
logger.setLevel(logging.DEBUG)


class FashionMNISTDataModule(L.LightningDataModule):
    CLASSES = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }

    def __init__(
        self,
        data_dir: Path = Path(__file__).parents[3].joinpath(".data/fashion_mnist_dataset"),
        batch_size: int = 32,
        num_workers: int = 4,
        prefetch_factor: int = 4,
        train_val_split: float = 0.8,
        horizontal_flip_prob: float = 0.5,
        rotation_degrees: int = 10,
        local_experiment: bool = False,
    ) -> None:

        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.train_val_split = train_val_split
        self.local_experiment = local_experiment

        # Tunable data augmentation hyperparameters
        self.horizontal_flip_prob = horizontal_flip_prob
        self.rotation_degrees = rotation_degrees

    def prepare_data(self) -> None:
        if self.local_experiment:
            logger.info(f"Downloading FashionMNIST data to {self.data_dir}.")
            FashionMNIST(root=self.data_dir, train=True, download=True)
            FashionMNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage: str) -> None:
        del stage  # Unused

        logger.info("Setting up FashionMNIST train/val/test datasets.")

        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.RandomHorizontalFlip(p=self.horizontal_flip_prob),
                transforms.RandomRotation(degrees=self.rotation_degrees),
            ]
        )
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        self.train_dataset = FashionMNIST(root=self.data_dir, train=True, download=False, transform=train_transform)
        train_size = int(len(self.train_dataset) * self.train_val_split)
        val_size = len(self.train_dataset) - train_size

        self.train_dataset, self.val_dataset = random_split(dataset=self.train_dataset, lengths=[train_size, val_size])
        self.test_dataset = FashionMNIST(root=self.data_dir, train=False, download=False, transform=test_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,
            shuffle=False,
        )

    def visualize_data(self, data_loader: DataLoader) -> None:
        # Get a batch of images and labels
        images, labels = next(iter(data_loader))

        # Set up a 4x4 grid for the images
        _, axes = plt.subplots(4, 4, figsize=(8, 8))

        # Display 16 images
        for i in range(16):
            ax = axes[i // 4, i % 4]

            # Convert the image from (C,H,W) to (H , W ,C)
            img = images[i].permute(1, 2, 0)

            ax.imshow(img, cmap="Greys_r")
            ax.set_title(f"Label: {self.CLASSES[labels[i].item()]}")
            ax.axis("off")

        plt.show()
