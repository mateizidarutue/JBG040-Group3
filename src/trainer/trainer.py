import optuna
import torch
from torch.utils.data import DataLoader
from torch import Tensor
import numpy as np
from typing import Dict, Any, Tuple, Optional
from optuna import Trial
from tqdm.auto import tqdm

from src.models.cnn import CNN
from src.factories.loss_factory import LossFactory
from src.factories.optimizer_factory import OptimizerFactory
from src.factories.scheduler_factory import SchedulerFactory
from src.factories.augmentation_factory import AugmentationFactory
from src.utils.metrics_calculator import MetricsCalculator
from src.types.evaluation_metrics import EvaluationMetrics
from src.utils.model_saver import ModelSaver
from src.types.train_return_type import TrainReturnType
from src.types.trial_type import TrialType


class Trainer:
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: str,
        input_size: int,
        num_classes: int,
    ) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.metrics_calculator = MetricsCalculator()
        self.input_size = input_size
        self.num_classes = num_classes

    def train(
        self, 
        params: Dict[str, Any], 
        num_epochs: int, 
        return_type: TrainReturnType = TrainReturnType.SCORE, 
        trial: Optional[Trial] = None
    ) -> float:
        model = CNN(params, self.num_classes, self.input_size).to(self.device)
        loss_fn = LossFactory.get_loss(params, self.num_classes).to(self.device)
        optimizer = OptimizerFactory.get_optimizer(model, params)
        scheduler = SchedulerFactory.get_scheduler(optimizer, params)
        augmentation = AugmentationFactory.get_augmentations(params, self.input_size)

        model.train()

        desc = f"Trial {trial.number} Epochs" if trial else "Training Epochs"
        epoch_pbar = tqdm(
            range(1, num_epochs + 1), desc=desc, leave=True
        )
        for epoch in epoch_pbar:
            epoch_losses = []

            batch_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
            for images, labels in batch_pbar:
                images: Tensor = images.to(self.device)
                images = augmentation(images)
                images = images.float()
                labels: Tensor = labels.to(self.device)
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

                batch_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = np.mean(epoch_losses)
            val_loss, metrics = self.test_val(model, params, True)

            epoch_pbar.set_postfix(
                {"train_loss": f"{avg_loss:.4f}", "val_loss": f"{val_loss:.4f}", "score": f"{metrics.score:.4f}"}
            )

            if trial:
                trial.report(metrics.score, step=epoch)

                if trial.should_prune():
                    test_loss, metrics = self.test_test(model, params, True)

                    ModelSaver.save_model(
                            model=model,
                            trial_type=TrialType.PRUNED,
                            params=params,
                            test_loss=test_loss,
                            metrics=metrics,
                            trial=trial,
                        )
                    
                    raise optuna.exceptions.TrialPruned()

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        test_loss, metrics = self.test_test(model, params, True)

        ModelSaver.save_model(
                model=model,
                trial_type=TrialType.COMPLETED,
                params=params,
                test_loss=test_loss,
                metrics=metrics,
                trial=trial,
            )
        self._model = model

        @property
        def model(self):
            return self._model

        match return_type:
            case TrainReturnType.SCORE:
                result = metrics.score
            case TrainReturnType.VAL_LOSS:
                result = val_loss
            case TrainReturnType.TEST_LOSS:
                result = test_loss

        return result

    def test_val(
        self,
        model: CNN,
        params: Dict[str, Any],
        calculate_metrics: bool,
    ) -> Tuple[float, Optional[EvaluationMetrics]]:
        return self.test(model, self.val_loader, params, calculate_metrics)

    def test_test(
        self,
        model: CNN,
        params: Dict[str, Any],
        calculate_metrics: bool,
    ) -> Tuple[float, Optional[EvaluationMetrics]]:
        return self.test(model, self.test_loader, params, calculate_metrics)

    def test(
        self,
        model: CNN,
        data_loader: DataLoader,
        params: Dict[str, Any],
        calculate_metrics: bool,
    ) -> Tuple[float, Optional[EvaluationMetrics]]:
        model.eval()
        total_loss = []
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in data_loader:
                images: Tensor = images.to(self.device)
                labels: Tensor = labels.to(self.device)
                outputs: Tensor = model(images)
                loss: Tensor = LossFactory.get_loss(params, self.num_classes)(
                    outputs, labels
                ).to(self.device)
                total_loss.append(loss.item())

                all_outputs.append(outputs)
                all_labels.append(labels)

        avg_loss = np.mean(total_loss)

        if calculate_metrics:
            all_outputs = torch.cat(all_outputs).detach().cpu()
            all_labels = torch.cat(all_labels).detach().cpu()
            metrics = self.metrics_calculator.compute(all_outputs, all_labels)
            model.train()
            return avg_loss, metrics
        else:
            model.train()
            return avg_loss, None
