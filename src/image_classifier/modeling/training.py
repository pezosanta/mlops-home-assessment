import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger("TRAINING")
logger.setLevel(logging.INFO)


def train(
    trainloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: str,
) -> None:

    torch_device = torch.device(device=device)

    logger.info(msg=f"Starting training for {num_epochs} epochs on {torch_device.type}.")
    for epoch in range(num_epochs):
        running_loss = 0.0

        # Using "uninterruptible" tqdm as progress bar.
        # All terminal logs within a tqdm progress will be printed above the progress bar.
        pbar = tqdm(unit="batches", total=len(trainloader), desc=f"Training epoch[{epoch+1}/{num_epochs}]", position=0)
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(torch_device), data[1].to(torch_device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                pbar.write(s="[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

            pbar.update()

    pbar.close()

    logger.info(msg="Finished training.")
