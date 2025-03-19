import optuna
from typing import Dict, List, Any
from src.search.param_sampler import ParamSampler
from torch.utils.data import DataLoader
from src.trainer.trainer import Trainer


class OptunaBayesHyperband:
    def __init__(
        self,
        min_budget: int,
        max_budget: int,
        eta: int,
        trials_per_search: int,
        searches_number: int,
        num_classes: int,
        trainer: Trainer,
        test_loader: DataLoader,
        config: Dict[str, Any],
        study_name: str = "hyperparameter_search",
        direction: str = "minimize",
    ):
        self.config = config
        self.trials_per_search = trials_per_search
        self.searches_number = searches_number
        self.num_classes = num_classes
        self.trainer = trainer
        self.test_loader = test_loader
        self.save_result: List[Dict[str, float]] = []

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

    def objective(self, trial: optuna.Trial):
        if trial.number < self.trials_per_search:
            params = ParamSampler.sample_random_params(self.config)
        else:
            params = ParamSampler.suggest_params(trial, self.config)

        result, trial_histories, model = self.trainer.train(trial, params)

        test_loss, metrics = self.trainer.test(model, self.test_loader, params, True)

        state = self.study.trials[trial.number].state

        self.save_result.append(
            {
                "trial_number": trial.number,
                "state": str(state),
                "params": params,
                "test_loss": test_loss,
                "metrics": metrics,
                "trial_histories": trial_histories,
            }
        )
        return result

    def run(self):
        total_trials = self.trials_per_search * self.searches_number
        print(f"Starting optimization with {total_trials} total trials...")

        for batch in range(self.searches_number):
            print(f"\n--- Batch {batch + 1}/{self.searches_number} ---")
            self._run_batch(self.trials_per_search)

    def _run_batch(self, batch_size: int):
        print(f"\n=== Running Batch of Size {batch_size} ===")

        for _ in range(batch_size):
            trial = self.study.ask()

            try:
                self.study.tell(trial, self.objective(trial))
            except optuna.exceptions.TrialPruned as e:
                print(f"Trial {trial.number} pruned: {e}")

        print(f"\n=== Finished Batch of Size {batch_size} ===")
