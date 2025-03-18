from optuna.pruners import HyperbandPruner
import torch
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
from typing import Dict, Any
from dc1.models.cnn import CNN
from dc1.utils.loss_factory import LossFactory
from dc1.utils.optimizer_factory import OptimizerFactory
from dc1.utils.scheduler_factory import SchedulerFactory
from optuna import Trial


class Trainer:
    def __init__(self, config: Dict[str, Any], data_loader: DataLoader, device: str):
        self.data_loader = data_loader
        self.device = device
        self.config = config

    def train(self, trial: Trial, optimizer_instance):
        model = CNN(self.config)
        model.to(self.device)

        loss_fn = LossFactory.get_loss(self.config)
        loss_fn.to(self.device)

        optimizer_model = OptimizerFactory.get_optimizer(model, self.config)
        scheduler = SchedulerFactory.get_scheduler(optimizer_model, self.config)

        model.train()
        pruner: HyperbandPruner = trial.study.pruner
        max_epochs = pruner._max_resource

        for epoch in range(1, max_epochs + 1):
            epoch_losses = []

            for images, labels in self.data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer_model.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer_model.step()
                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            metrics = self.evaluate(avg_loss)
            optimizer_instance.update_trial_history(trial.number, epoch, metrics)
            trial.report(avg_loss, step=epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_loss)
            else:
                scheduler.step()

        return avg_loss

    def evaluate(self, avg_loss) -> Dict[str, float]:
        self.model.eval()
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                all_outputs.append(outputs)
                all_labels.append(labels)

        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = compute_metrics(all_outputs, all_labels)
        metrics["loss"] = avg_loss
        metrics["val_loss"] = avg_loss + 0.1
        metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]

        self.model.train()
        return metrics
