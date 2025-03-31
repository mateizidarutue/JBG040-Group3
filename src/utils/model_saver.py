from optuna import Trial
from typing import Dict
from torch.nn import Module
import torch
import json
import os

from src.types.evaluation_metrics import EvaluationMetrics
from src.types.trial_type import TrialType

class ModelSaver:
    @staticmethod
    def save_model(
        model: Module,
        trial: Trial,
        trial_type: TrialType,
        params: Dict,
        test_loss: float,
        metrics: EvaluationMetrics,
    ) -> None:
        if trial:
            trial.set_user_attr("test_loss", test_loss)
            trial.set_user_attr("metrics", metrics.to_dict())
            trial_number = trial.number
        else:
            trial_number = "manual"

        dir_path = f"{trial_type.value}/model_trial_{trial_number}"

        print(f"Saving model to {dir_path}...")
        os.makedirs(dir_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(dir_path, "model.pt"))

        metadata = {
            "trial_number": trial.number,
            "params": params,
            "test_loss": test_loss,
            "metrics": metrics.to_dict(),
        }

        with open(os.path.join(dir_path, "info.json"), "w") as f:
            json.dump(metadata, f, indent=4)
