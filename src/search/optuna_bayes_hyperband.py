import optuna
from optuna.study import MaxTrialsCallback
from typing import Dict, Any
import os
from dotenv import load_dotenv

from src.search.param_sampler import ParamSampler
from src.trainer.trainer import Trainer


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
        config: Dict[str, Any],
        study_name: str = "hyperparameter_search",
        storage_connection_string: str = os.getenv("STORAGE_CONNECTION_STRING"),
    ):
        self.config = config
        self.total_trials = total_trials
        self.num_classes = num_classes
        self.trainer = trainer
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
            direction="minimize",
            sampler=self.sampler,
            load_if_exists=True,
            pruner=self.pruner,
        )

    def objective(self, trial: optuna.Trial) -> float:
        params = ParamSampler.suggest_params(trial, self.config)

        combined_score = self.trainer.train(params, self.num_epochs, trial)

        return combined_score

    def run(self):
        print(f"Starting optimization with {self.total_trials} total trials...")
        self.study.optimize(
            self.objective,
            callbacks=[MaxTrialsCallback(self.total_trials, states=None)],
        )
