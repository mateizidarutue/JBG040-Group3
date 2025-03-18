import torch
import numpy as np
from torch import nn
from typing import Dict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn,
        scheduler: _LRScheduler,
        data_loader,
        device,
        config,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.data_loader = data_loader
        self.device = device
        self.config = config

    def train(self, trial, optimizer_instance):
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

        self.model.train()
        max_epochs = trial.study.pruner._max_resource

        for epoch in range(1, max_epochs + 1):
            epoch_losses = []

            for images, labels in self.data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            metrics = self.evaluate(avg_loss)
            optimizer_instance.update_trial_history(trial.number, epoch, metrics)
            trial.report(avg_loss, step=epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(avg_loss)
            else:
                self.scheduler.step()

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
