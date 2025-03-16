import random
import json
import optuna
from optuna.pruners import HyperbandPruner
from typing import Callable, Dict, Any


class Search:
    def __init__(
        self,
        config: Dict[str, Any],
        train_fn: Callable,
        num_random_trials: int = 10,
        n_trials: int = 50,
    ):
        self.config = config
        self.train_fn = train_fn
        self.num_random_trials = num_random_trials
        self.n_trials = n_trials
        self.study = optuna.create_study(direction="minimize", pruner=HyperbandPruner())
        self._initialize_random_search()

    def _initialize_random_search(self):
        print(f"Running {self.num_random_trials} random search trials...")
        for _ in range(self.num_random_trials):
            params = self._sample_params(random_sample=True)
            self.study.enqueue_trial(params)

    def _sample_params(self, random_sample: bool = False) -> Dict[str, Any]:
        if random_sample:
            sample_fn = lambda x: (
                random.choice(x)
                if isinstance(x, list)
                else random.uniform(x["min"], x["max"])
            )
        else:
            sample_fn = lambda x, name: (
                self.trial.suggest_categorical(name, x)
                if isinstance(x, list)
                else self.trial.suggest_float(name, x["min"], x["max"])
            )

        cfg = self.config
        params = {
            "conv_layers": (
                sample_fn(cfg["architecture"]["conv_layers"], "conv_layers")
                if not random_sample
                else sample_fn(cfg["architecture"]["conv_layers"])
            ),
            "kernel_size": (
                sample_fn(cfg["architecture"]["kernel_size"], "kernel_size")
                if not random_sample
                else sample_fn(cfg["architecture"]["kernel_size"])
            ),
            "stride": (
                sample_fn(cfg["architecture"]["stride"], "stride")
                if not random_sample
                else sample_fn(cfg["architecture"]["stride"])
            ),
            "padding": (
                sample_fn(cfg["architecture"]["padding"], "padding")
                if not random_sample
                else sample_fn(cfg["architecture"]["padding"])
            ),
            "use_batch_norm": (
                sample_fn(cfg["architecture"]["use_batch_norm"], "use_batch_norm")
                if not random_sample
                else sample_fn(cfg["architecture"]["use_batch_norm"])
            ),
            "use_layer_norm": (
                sample_fn(cfg["architecture"]["use_layer_norm"], "use_layer_norm")
                if not random_sample
                else sample_fn(cfg["architecture"]["use_layer_norm"])
            ),
            "use_group_norm": (
                sample_fn(cfg["architecture"]["use_group_norm"], "use_group_norm")
                if not random_sample
                else sample_fn(cfg["architecture"]["use_group_norm"])
            ),
            "use_instance_norm": (
                sample_fn(cfg["architecture"]["use_instance_norm"], "use_instance_norm")
                if not random_sample
                else sample_fn(cfg["architecture"]["use_instance_norm"])
            ),
            "min_max_scaling": (
                sample_fn(cfg["architecture"]["min_max_scaling"], "min_max_scaling")
                if not random_sample
                else sample_fn(cfg["architecture"]["min_max_scaling"])
            ),
            "activation": (
                sample_fn(cfg["architecture"]["activation"], "activation")
                if not random_sample
                else sample_fn(cfg["architecture"]["activation"])
            ),
            "weight_initialization": (
                sample_fn(
                    cfg["architecture"]["weight_initialization"],
                    "weight_initialization",
                )
                if not random_sample
                else sample_fn(cfg["architecture"]["weight_initialization"])
            ),
            "freeze_layers": (
                sample_fn(cfg["architecture"]["freeze_layers"], "freeze_layers")
                if not random_sample
                else sample_fn(cfg["architecture"]["freeze_layers"])
            ),
            "hidden_layers": (
                sample_fn(cfg["fully_connected"]["hidden_layers"], "hidden_layers")
                if not random_sample
                else sample_fn(cfg["fully_connected"]["hidden_layers"])
            ),
            "fc_activation": (
                sample_fn(cfg["fully_connected"]["activation"], "fc_activation")
                if not random_sample
                else sample_fn(cfg["fully_connected"]["activation"])
            ),
            "learning_rate": (
                sample_fn(cfg["training"]["learning_rate"], "learning_rate")
                if not random_sample
                else sample_fn(cfg["training"]["learning_rate"])
            ),
            "momentum": (
                sample_fn(cfg["training"]["momentum"], "momentum")
                if not random_sample
                else sample_fn(cfg["training"]["momentum"])
            ),
            "weight_decay": (
                sample_fn(cfg["training"]["weight_decay"], "weight_decay")
                if not random_sample
                else sample_fn(cfg["training"]["weight_decay"])
            ),
            "weight_decay_on_bias": (
                sample_fn(
                    cfg["training"]["weight_decay_on_bias"], "weight_decay_on_bias"
                )
                if not random_sample
                else sample_fn(cfg["training"]["weight_decay_on_bias"])
            ),
            "gradient_clipping": (
                sample_fn(cfg["training"]["gradient_clipping"], "gradient_clipping")
                if not random_sample
                else sample_fn(cfg["training"]["gradient_clipping"])
            ),
            "optimizer": (
                sample_fn(cfg["training"]["optimizer"], "optimizer")
                if not random_sample
                else sample_fn(cfg["training"]["optimizer"])
            ),
            "epochs": (
                sample_fn(cfg["training"]["epochs"], "epochs")
                if not random_sample
                else sample_fn(cfg["training"]["epochs"])
            ),
            "scheduler": (
                sample_fn(cfg["training"]["scheduler"], "scheduler")
                if not random_sample
                else sample_fn(cfg["training"]["scheduler"])
            ),
            "decay_factor": (
                sample_fn(cfg["training"]["decay_factor"], "decay_factor")
                if not random_sample
                else sample_fn(cfg["training"]["decay_factor"])
            ),
            "warmup_steps": (
                sample_fn(cfg["training"]["warmup_steps"], "warmup_steps")
                if not random_sample
                else sample_fn(cfg["training"]["warmup_steps"])
            ),
            "early_stopping": {
                "enabled": (
                    sample_fn(
                        cfg["training"]["early_stopping"]["enabled"],
                        "early_stopping_enabled",
                    )
                    if not random_sample
                    else sample_fn(cfg["training"]["early_stopping"]["enabled"])
                ),
                "patience": (
                    sample_fn(
                        cfg["training"]["early_stopping"]["patience"],
                        "early_stopping_patience",
                    )
                    if not random_sample
                    else sample_fn(cfg["training"]["early_stopping"]["patience"])
                ),
            },
            "shuffle": (
                sample_fn(cfg["training"]["shuffle"], "shuffle")
                if not random_sample
                else sample_fn(cfg["training"]["shuffle"])
            ),
            "regularization": {
                "use_l1": (
                    sample_fn(cfg["regularization"]["use_l1"], "use_l1")
                    if not random_sample
                    else sample_fn(cfg["regularization"]["use_l1"])
                ),
                "l1_strength": (
                    sample_fn(cfg["regularization"]["l1_strength"], "l1_strength")
                    if not random_sample
                    else sample_fn(cfg["regularization"]["l1_strength"])
                ),
                "use_l2": (
                    sample_fn(cfg["regularization"]["use_l2"], "use_l2")
                    if not random_sample
                    else sample_fn(cfg["regularization"]["use_l2"])
                ),
                "l2_strength": (
                    sample_fn(cfg["regularization"]["l2_strength"], "l2_strength")
                    if not random_sample
                    else sample_fn(cfg["regularization"]["l2_strength"])
                ),
                "use_dropout": (
                    sample_fn(cfg["regularization"]["use_dropout"], "use_dropout")
                    if not random_sample
                    else sample_fn(cfg["regularization"]["use_dropout"])
                ),
                "dropout_rate": (
                    sample_fn(cfg["regularization"]["dropout_rate"], "dropout_rate")
                    if not random_sample
                    else sample_fn(cfg["regularization"]["dropout_rate"])
                ),
                "dropout_position": (
                    sample_fn(
                        cfg["regularization"]["dropout_position"], "dropout_position"
                    )
                    if not random_sample
                    else sample_fn(cfg["regularization"]["dropout_position"])
                ),
                "use_dropconnect": (
                    sample_fn(
                        cfg["regularization"]["use_dropconnect"], "use_dropconnect"
                    )
                    if not random_sample
                    else sample_fn(cfg["regularization"]["use_dropconnect"])
                ),
                "dropconnect_rate": (
                    sample_fn(
                        cfg["regularization"]["dropconnect_rate"], "dropconnect_rate"
                    )
                    if not random_sample
                    else sample_fn(cfg["regularization"]["dropconnect_rate"])
                ),
            },
            "augmentation": {
                "use_brightness": (
                    sample_fn(cfg["augmentation"]["use_brightness"], "use_brightness")
                    if not random_sample
                    else sample_fn(cfg["augmentation"]["use_brightness"])
                ),
                "brightness": (
                    sample_fn(cfg["augmentation"]["brightness"], "brightness")
                    if not random_sample
                    else sample_fn(cfg["augmentation"]["brightness"])
                ),
                "use_contrast": (
                    sample_fn(cfg["augmentation"]["use_contrast"], "use_contrast")
                    if not random_sample
                    else sample_fn(cfg["augmentation"]["use_contrast"])
                ),
                "contrast": (
                    sample_fn(cfg["augmentation"]["contrast"], "contrast")
                    if not random_sample
                    else sample_fn(cfg["augmentation"]["contrast"])
                ),
                "use_flip": (
                    sample_fn(cfg["augmentation"]["use_flip"], "use_flip")
                    if not random_sample
                    else sample_fn(cfg["augmentation"]["use_flip"])
                ),
                "flip": (
                    sample_fn(cfg["augmentation"]["flip"], "flip")
                    if not random_sample
                    else sample_fn(cfg["augmentation"]["flip"])
                ),
                "use_crop": (
                    sample_fn(cfg["augmentation"]["use_crop"], "use_crop")
                    if not random_sample
                    else sample_fn(cfg["augmentation"]["use_crop"])
                ),
                "crop": (
                    sample_fn(cfg["augmentation"]["crop"], "crop")
                    if not random_sample
                    else sample_fn(cfg["augmentation"]["crop"])
                ),
                "use_upscale": (
                    sample_fn(cfg["augmentation"]["use_upscale"], "use_upscale")
                    if not random_sample
                    else sample_fn(cfg["augmentation"]["use_upscale"])
                ),
                "upscale": (
                    sample_fn(cfg["augmentation"]["upscale"], "upscale")
                    if not random_sample
                    else sample_fn(cfg["augmentation"]["upscale"])
                ),
                "upscale_mode": (
                    sample_fn(cfg["augmentation"]["upscale_mode"], "upscale_mode")
                    if not random_sample
                    else sample_fn(cfg["augmentation"]["upscale_mode"])
                ),
            },
        }
        return params

    def _objective(self, trial):
        self.trial = trial
        params = self._sample_params()

        metrics_to_track = self.config["evaluation"]["metrics"]

        metrics = self.train_fn(params, trial)

        for metric in metrics_to_track:
            if metric not in metrics:
                raise ValueError(f"Metric {metric} not found in training results")

        primary_metric = metrics.get("val_loss", metrics[metrics_to_track[0]])

        return primary_metric

    def run_search(self):
        print(f"Running Bayesian Search + Hyperband with {self.n_trials} trials...")
        self.study.optimize(self._objective, n_trials=self.n_trials)
        self._save_results()

    def _save_results(self):
        print("\nBest Trial:")
        trial_dict = {
            "params": self.study.best_trial.params,
            "value": self.study.best_trial.value,
            "metrics": self.study.best_trial.user_attrs.get("metrics", {}),
        }
        print(json.dumps(trial_dict, indent=2))

        trials_df = self.study.trials_dataframe()
        results = {
            "best_trial": trial_dict,
            "all_trials": trials_df.to_dict(),
            "evaluation_metrics": self.config["evaluation"]["metrics"],
        }

        with open("search_results.json", "w") as f:
            json.dump(results, f, indent=4)

        print("Results saved to search_results.json")
