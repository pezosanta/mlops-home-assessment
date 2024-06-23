import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger("VALIDATION")
logger.setLevel(logging.INFO)


def validation(testloader: DataLoader, model: nn.Module, classes: list[str], device: str) -> dict[str, float]:

    device = torch.device(device=device)

    @torch.no_grad()
    def get_total_accuracy() -> float:
        correct = 0
        total = 0

        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        logger.info(msg=f"Total test accuracy: {accuracy:.2f} %")

        return accuracy

    @torch.no_grad()
    def get_classwise_accuracy() -> dict[str, float]:
        accuracy = dict()

        class_correct = list(0.0 for i in range(len(classes)))
        class_total = list(0.0 for i in range(len(classes)))

        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        for i, curr_class in enumerate(classes):
            curr_accuracy = 100 * class_correct[i] / class_total[i]
            accuracy.update({curr_class: curr_accuracy})

            logger.info(msg=f"Class-wise accuracy [{curr_class}]: {curr_accuracy:.2f} %")

        return accuracy

    logger.info(
        msg=f"Starting validation/testing on the {len(testloader)*testloader.batch_size} validation/test images."
    )
    class_wise_accuracy = get_classwise_accuracy()
    class_wise_accuracy.update({"total": get_total_accuracy()})

    logger.info(msg="Finished validation/testing.")

    return class_wise_accuracy
