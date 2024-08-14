import logging

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics

logger = logging.getLogger("ORCHESTRATOR")
logger.setLevel(logging.INFO)


class Orchestrator(L.LightningModule):
    def __init__(self, model: nn.Module, classes: dict[int, str], learning_rate: float = 0.001) -> None:

        super().__init__()

        self.classes = classes
        self.learning_rate = learning_rate

        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.overall_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=len(self.classes), average="macro")
        self.overall_precision = torchmetrics.Precision(
            task="multiclass", num_classes=len(self.classes), average="macro"
        )
        self.overall_recall = torchmetrics.Recall(task="multiclass", num_classes=len(self.classes), average="macro")
        self.overall_f1_score = torchmetrics.F1Score(task="multiclass", num_classes=len(self.classes), average="macro")

        # As of lightning>=2.0.0 you have to define the train/validation/test_step_outputs in the init clause
        # of your module to store the train/validation/test outputs of each epoch.
        self.train_step_outputs: list[dict[str, list[torch.Tensor]]] = []
        self.validation_step_outputs: list[dict[str, list[torch.Tensor]]] = []
        self.test_step_outputs: list[dict[str, list[torch.Tensor]]] = []

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(params=self.model.parameters(), lr=self.learning_rate)

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx  # Unused

        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        self.log(name="train_loss", value=loss, on_step=False, on_epoch=True, prog_bar=True)

        self.train_step_outputs.append({"train_loss": loss, "predictions": outputs, "labels": labels})

        return loss

    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx  # Unused

        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        self.log(name="val_loss", value=loss, on_step=False, on_epoch=True, prog_bar=True)

        self.validation_step_outputs.append({"val_loss": loss, "predictions": outputs, "labels": labels})

        return loss

    def test_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx  # Unused

        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        self.log(name="test_loss", value=loss, on_step=False, on_epoch=True, prog_bar=True)

        self.test_step_outputs.append({"test_loss": loss, "predictions": outputs, "labels": labels})

        return loss

    def on_train_epoch_end(self) -> None:
        predictions = torch.cat([x["predictions"] for x in self.train_step_outputs])
        labels = torch.cat([x["labels"] for x in self.train_step_outputs])

        # Overall metrics
        self.log_dict(
            {
                "train_accuracy": self.overall_accuracy(predictions, labels),
                "train_precision": self.overall_precision(predictions, labels),
                "train_recall": self.overall_recall(predictions, labels),
                "train_f1_score": self.overall_f1_score(predictions, labels),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.train_step_outputs.clear()  # free memory

    def on_validation_epoch_end(self) -> None:
        predictions = torch.cat([x["predictions"] for x in self.validation_step_outputs])
        labels = torch.cat([x["labels"] for x in self.validation_step_outputs])

        # Overall metrics
        self.log_dict(
            {
                "val_accuracy": self.overall_accuracy(predictions, labels),
                "val_precision": self.overall_precision(predictions, labels),
                "val_recall": self.overall_recall(predictions, labels),
                "val_f1_score": self.overall_f1_score(predictions, labels),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.validation_step_outputs.clear()  # free memory

    def on_test_epoch_end(self) -> None:
        predictions = torch.cat([x["predictions"] for x in self.test_step_outputs])
        labels = torch.cat([x["labels"] for x in self.test_step_outputs])

        # Overall metrics
        self.log_dict(
            {
                "test_accuracy": self.overall_accuracy(predictions, labels),
                "test_precision": self.overall_precision(predictions, labels),
                "test_recall": self.overall_recall(predictions, labels),
                "test_f1_score": self.overall_f1_score(predictions, labels),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.test_step_outputs.clear()  # free memory

    # def on_test_end(self):
    #     import mlflow
    #     # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    #     # self.logger.experiment.experiment_name = "fashion_mnist_test_1"
    #     # mlflow.set_experiment("fashion_mnist_test_1")
    #     mlflow.pytorch.log_model(self.model, artifact_path="model")
    #     # self.logger.experiment.log_model(self.model, artifact_path="model")
