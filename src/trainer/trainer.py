from optuna.pruners import HyperbandPruner
import optuna
import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, List
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
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
        input_size: int,
        num_classes: int,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.metrics_calculator = MetricsCalculator()
        self.input_size = input_size
        self.num_classes = num_classes
        self.trial_histories: List[Dict[str, float]] = []

    def train(self, trial: Trial, params: Dict[str, Any]):
        model = CNN(params, self.num_classes, self.input_size).to(self.device)
        loss_fn = LossFactory.get_loss(params).to(self.device)
        optimizer = OptimizerFactory.get_optimizer(model, params)
        scheduler = SchedulerFactory.get_scheduler(optimizer, params)
        augmentation = AugmentationFactory.get_augmentations(params, self.input_size)

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
                if params["gradient_clipping_enabled"]:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), params["gradient_clipping"]
                    )
                optimizer.step()
                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            val_loss, _ = self.test(model, self.val_loader, loss_fn, False)
            self.trial_histories.append(
                {
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "val_loss": val_loss,
                }
            )

            trial.report(avg_loss, step=epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        return val_loss, self.trial_histories, model

    def test(
        self,
        model: CNN,
        data_loader: DataLoader,
        params: Dict[str, Any],
        metrics: bool,
    ):
        model.eval()
        total_loss = []
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in data_loader:
                images: Tensor = images.to(self.device)
                labels: Tensor = labels.to(self.device)
                outputs: Tensor = model(images)
                loss: Tensor = LossFactory.get_loss(params)(outputs, labels)
                total_loss.append(loss.item())

                all_outputs.append(outputs)
                all_labels.append(labels)

        avg_loss = np.mean(total_loss)

        if metrics:
            all_outputs = torch.cat(all_outputs).detach().cpu()
            all_labels = torch.cat(all_labels).detach().cpu()
            metrics = self.metrics_calculator.compute(all_outputs, all_labels)
            return avg_loss, metrics
        else:
            return avg_loss, None
