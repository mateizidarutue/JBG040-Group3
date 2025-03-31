from optuna import Trial
from typing import Dict, Optional
from torch.nn import Module
import torch
import json
import os
import re

from src.types.evaluation_metrics import EvaluationMetrics
from src.types.trial_type import TrialType

class ModelSaver:
    @staticmethod
    def save_model(
        model: Module,
        trial_type: TrialType,
        params: Dict,
        test_loss: float,
        metrics: EvaluationMetrics,
        trial: Optional[Trial] = None,
    ) -> None:
        base_dir = os.path.join("saved_outputs", trial_type.value)
        os.makedirs(base_dir, exist_ok=True)

        if trial:
            trial.set_user_attr("test_loss", test_loss)
            trial.set_user_attr("metrics", metrics.to_dict())
            trial_number = trial.number
            
            dir_path = os.path.join(base_dir, f"trial_{trial_number}")
        else:
            existing = [
                d for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d)) and re.match(r"manual_\d+", d)
            ]
            indices = [
                int(re.search(r"manual_(\d+)", name).group(1))
                for name in existing
            ] if existing else []

            next_index = max(indices) + 1 if indices else 0
            dir_path = os.path.join(base_dir, f"manual_{next_index}")
            trial_number = f"manual_{next_index}"


        print(f"Saving model to {dir_path}...")
        os.makedirs(dir_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(dir_path, "model.pt"))

        metadata = {
            "trial_number": trial_number,
            "params": params,
            "test_loss": test_loss,
            "metrics": metrics.to_dict(),
        }

        with open(os.path.join(dir_path, "info.json"), "w") as f:
            json.dump(metadata, f, indent=4)
