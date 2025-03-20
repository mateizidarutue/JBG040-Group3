import optuna
from typing import Dict, List, Any

from optuna.study import MaxTrialsCallback
from src.search.param_sampler import ParamSampler
from torch.utils.data import DataLoader
from src.trainer.trainer import Trainer
import torch
import json
import os
from dotenv import load_dotenv

load_dotenv()


class OptunaBayesHyperband:
    def __init__(
        self,
        min_budget: int,
        max_budget: int,
        eta: int,
        total_trials: int,
        num_classes: int,
        trainer: Trainer,
        test_loader: DataLoader,
        config: Dict[str, Any],
        study_name: str = "hyperparameter_search",
        direction: str = "minimize",
        storage_connection_string: str = os.getenv("STORAGE_CONNECTION_STRING"),
    ):
        self.config = config
        self.total_trials = total_trials
        self.num_classes = num_classes
        self.trainer = trainer
        self.test_loader = test_loader
        self.save_result: List[Dict[str, float]] = []
        self.num_epochs = max_budget

        self.pruner = optuna.pruners.HyperbandPruner(
            min_resource=min_budget, max_resource=max_budget, reduction_factor=eta
        )
        self.sampler = optuna.samplers.TPESampler(
            multivariate=True,
            group=True,
            constant_liar=True,
            n_startup_trials=10,
        )
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage_connection_string,
            direction=direction,
            sampler=self.sampler,
            load_if_exists=True,
            pruner=self.pruner,
        )

    def objective(self, trial: optuna.Trial):
        params = ParamSampler.suggest_params(trial, self.config)

        result, model = self.trainer.train(trial, params, self.num_epochs)

        test_loss, metrics = self.trainer.test(model, self.test_loader, params, True)

        trial.set_user_attr("test_loss", test_loss)
        trial.set_user_attr("metrics", metrics)

        model_path = f"saved_models/model_trial_{trial.number}.pt"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)

        return result

    def run(self):
        print(f"Starting optimization with {self.total_trials} total trials...")
        self.study.optimize(
            self.objective,
            callbacks=[MaxTrialsCallback(self.total_trials, states=None)],
        )

        with open("results.json", "w") as f:
            json.dump(self.save_result, f, indent=4)
