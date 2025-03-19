import optuna
from typing import Dict, List, Any
import yaml
import json
from param_sampler import ParamSampler
import torch


class OptunaBayesHyperband:
    def __init__(
        self,
        min_budget: int,
        max_budget: int,
        eta: int,
        trials_per_batch: int,
        number_of_runs: int,
        training_fn,
        config: Dict[str, Any],
        study_name: str = "hyperparameter_search",
        direction: str = "minimize",
    ):
        self.config = config
        self.trials_per_batch = trials_per_batch
        self.number_of_runs = number_of_runs
        self.training_fn = training_fn

        self.trial_histories: Dict[int, Dict[str, List[float]]] = {}

        self.pruner = optuna.pruners.HyperbandPruner(
            min_resource=min_budget, max_resource=max_budget, reduction_factor=eta
        )
        self.sampler = optuna.samplers.TPESampler()
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=self.sampler,
            pruner=self.pruner,
        )

    def update_trial_history(
        self, trial_number: int, epoch: int, metrics: Dict[str, float]
    ):
        if trial_number not in self.trial_histories:
            self.trial_histories[trial_number] = {
                "epoch": [],
                "loss": [],
                "val_loss": [],
                "learning_rate": [],
            }
        self.trial_histories[trial_number]["epoch"].append(epoch)
        self.trial_histories[trial_number]["loss"].append(metrics["loss"])
        self.trial_histories[trial_number]["val_loss"].append(metrics["val_loss"])
        self.trial_histories[trial_number]["learning_rate"].append(
            metrics["learning_rate"]
        )

    def objective(self, trial: optuna.Trial):
        if trial.number < self.trials_per_batch:
            params = ParamSampler.sample_random_params(self.config)
        else:
            params = ParamSampler.suggest_params(trial, self.config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trial.set_user_attr("device", str(device))

        result = self.training_fn(trial, params, self)

        if trial.number in self.trial_histories:
            history = self.trial_histories[trial.number]
            if history["loss"]:
                final_loss = history["loss"][-1]
                final_val_loss = history["val_loss"][-1]
                trial.set_user_attr("final_loss", final_loss)
                trial.set_user_attr("final_val_loss", final_val_loss)

        return result

    def run(self):
        total_trials = self.trials_per_batch * self.number_of_runs
        print(f"Starting optimization with {total_trials} total trials...")

        for batch in range(self.number_of_runs):
            print(f"\n--- Batch {batch + 1}/{self.number_of_runs} ---")
            self._run_batch(self.trials_per_batch)

    def save_results(self, filename: str):
        results = []
        for trial in self.study.trials:
            trial_history = self.trial_histories.get(trial.number, {})
            record = {
                "trial_number": trial.number,
                "state": str(trial.state),
                "params": trial.params,
                "history": trial_history,
                "user_attrs": trial.user_attrs,
            }
            results.append(record)
        if self.direction == "minimize":
            results.sort(key=lambda r: r["user_attrs"].get("final_loss", float("inf")))
        else:
            results.sort(
                key=lambda r: r["user_attrs"].get("final_loss", float("-inf")),
                reverse=True,
            )
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)

    def _run_batch(self, batch_size: int):
        print(f"\n=== Running Batch of Size {batch_size} ===")

        for _ in range(batch_size):
            trial = self.study.ask()

            try:
                self.study.tell(trial, self.objective(trial))
            except optuna.exceptions.TrialPruned as e:
                print(f"Trial {trial.number} pruned: {e}")

        print(f"\n=== Finished Batch of Size {batch_size} ===")

    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
