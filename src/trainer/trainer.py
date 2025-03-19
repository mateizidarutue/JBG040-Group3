from optuna.pruners import HyperbandPruner
import optuna
import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any
from src.models.cnn import CNN
from src.utils.loss_factory import LossFactory
from src.utils.optimizer_factory import OptimizerFactory
from src.utils.scheduler_factory import SchedulerFactory
from src.utils.augmentation_factory import AugmentationFactory
from src.utils.metrics_calculator import MetricsCalculator
from optuna import Trial
from torch import Tensor


class Trainer:
    def __init__(
        self,
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.metrics_calculator = MetricsCalculator()

    def train(self, trial: Trial, optimizer_instance):
        model = CNN(self.config).to(self.device)
        loss_fn = LossFactory.get_loss(self.config).to(self.device)
        optimizer = OptimizerFactory.get_optimizer(model, self.config)
        scheduler = SchedulerFactory.get_scheduler(optimizer, self.config)
        augmentation = AugmentationFactory.get_augmentations(self.config)

        pruner: HyperbandPruner = trial.study.pruner
        max_epochs = pruner._max_resource

        model.train()

        for epoch in range(1, max_epochs + 1):
            epoch_losses = []

            for images, labels in self.train_loader:
                images: Tensor = images.to(self.device)
                labels: Tensor = labels.to(self.device)
                images = augmentation(images)
                optimizer.zero_grad()
                outputs: Tensor = model(images)
                loss: Tensor = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)

            val_loss, metrics = self.test(model, self.val_loader, loss_fn)
            optimizer_instance.update_trial_history(
                trial.number, epoch, {"loss": avg_loss, "val_loss": val_loss, **metrics}
            )

            trial.report(val_loss, step=epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        return val_loss

    def test(self, model: CNN, data_loader: DataLoader, loss_fn: torch.nn.Module):
        model.eval()
        total_loss = []
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in data_loader:
                images: Tensor = images.to(self.device)
                labels: Tensor = labels.to(self.device)
                outputs: Tensor = model(images)
                loss: Tensor = loss_fn(outputs, labels)
                total_loss.append(loss.item())

                all_outputs.append(outputs)
                all_labels.append(labels)

        avg_loss = np.mean(total_loss)

        all_outputs = torch.cat(all_outputs).detach().cpu()
        all_labels = torch.cat(all_labels).detach().cpu()

        metrics = self.metrics_calculator.compute(all_outputs, all_labels)

        return avg_loss, metrics
