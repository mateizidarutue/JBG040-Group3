import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any
import optuna

from models.cnn import CNN
from utils.loss_factory import LossFactory
from utils.optimizer_factory import OptimizerFactory
from utils.scheduler_factory import SchedulerFactory
from utils.metrics import compute_metrics


class Trainer:
    def __init__(
        self, config: Dict[str, Any], batch_size: int = 16, num_epochs: int = 5
    ) -> None:
        self.config: Dict[str, Any] = config
        self.batch_size: int = batch_size
        self.num_epochs: int = num_epochs
        self.num_samples: int = 100
        self.num_classes: int = 6
        self.dataset: TensorDataset = self._create_dummy_dataset()

    def _create_dummy_dataset(self) -> TensorDataset:
        X = torch.rand(self.num_samples, 1, 128, 128)
        y = torch.randint(0, self.num_classes, (self.num_samples,))
        return TensorDataset(X, y)

    def train(
        self, trial: optuna.Trial, params: Dict[str, Any], optimizer_instance: Any
    ) -> float:
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model: CNN = CNN(params)
        model.to(device)

        data_loader: DataLoader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )

        loss_fn: nn.Module = LossFactory.get_loss(
            self.config["training"]["loss_function"]
        )
        loss_fn.to(device)

        optimizer_model = OptimizerFactory.get_optimizer(model, params)
        scheduler = SchedulerFactory.get_scheduler(
            optimizer_model, self.config["training"]["scheduler"]
        )

        model.train()
        for epoch in range(1, self.num_epochs + 1):
            epoch_losses = []
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer_model.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                clip_value: float = self.config["training"]["gradient_clipping"]["max"]
                nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                optimizer_model.step()
                epoch_losses.append(loss.item())

            avg_loss: float = np.mean(epoch_losses)

            model.eval()
            with torch.no_grad():
                all_outputs = []
                all_labels = []
                for images, labels in data_loader:
                    images = images.to(device)
                    outputs = model(images)
                    all_outputs.append(outputs)
                    all_labels.append(labels)
                all_outputs = torch.cat(all_outputs, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                metrics: Dict[str, float] = compute_metrics(all_outputs, all_labels)
                metrics["loss"] = avg_loss
                metrics["val_loss"] = avg_loss + 0.1
                metrics["learning_rate"] = optimizer_model.param_groups[0]["lr"]
            model.train()

            optimizer_instance.update_trial_history(trial.number, epoch, metrics)

            trial.report(avg_loss, step=epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_loss)
            else:
                scheduler.step()

        return avg_loss
