import random
from typing import Dict, Any
import optuna


class ParamSampler:
    @staticmethod
    def sample_random_params(config: Dict) -> Dict[str, Any]:
        params = {}

        arch = config["architecture"]
        params.update(
            {
                "conv_layers": random.choice(arch["conv_layers"]),
                "kernel_size": random.choice(arch["kernel_size"]),
                "stride": random.choice(arch["stride"]),
                "padding": random.choice(arch["padding"]),
                "use_batch_norm": random.choice(arch["use_batch_norm"]),
                "use_group_norm": random.choice(arch["use_group_norm"]),
                "use_instance_norm": random.choice(arch["use_instance_norm"]),
                "use_min_max_scaling": random.choice(arch["use_min_max_scaling"]),
                "activation_conv": random.choice(arch["activation_conv"]),
                "weight_initialization": random.choice(arch["weight_initialization"]),
                "freeze_layers": random.choice(arch["freeze_layers"]),
                "fully_connected_layers": random.choice(arch["fully_connected_layers"]),
                "activation_fc": random.choice(arch["activation_fc"]),
            }
        )

        train = config["training"]
        params.update(
            {
                "gradient_clipping": random.uniform(
                    train["gradient_clipping"]["min"], train["gradient_clipping"]["max"]
                ),
                "early_stopping_enabled": random.choice(
                    train["early_stopping"]["enabled"]
                ),
                "early_stopping_patience": random.choice(
                    train["early_stopping"]["patience"]
                ),
            }
        )

        opt = train["optimizer"]
        params.update(
            {
                "optimizer": random.choice(opt["type"]),
                "learning_rate": random.uniform(
                    opt["learning_rate"]["min"], opt["learning_rate"]["max"]
                ),
                "momentum": random.uniform(
                    opt["momentum"]["min"], opt["momentum"]["max"]
                ),
                "weight_decay": random.uniform(
                    opt["weight_decay"]["min"], opt["weight_decay"]["max"]
                ),
                "weight_decay_on_bias": random.choice(opt["weight_decay_on_bias"]),
            }
        )

        scheduler = train["scheduler"]
        params.update(
            {
                "scheduler": random.choice(scheduler["type"]),
                "lr_decay_factor": random.uniform(
                    scheduler["lr_decay_factor"]["min"],
                    scheduler["lr_decay_factor"]["max"],
                ),
                "step_size": random.choice(scheduler["step_size"]),
            }
        )

        loss = train["loss"]
        params["loss"] = {
            "type": random.choice(loss["types"]),
            "gamma": random.uniform(loss["gamma"]["min"], loss["gamma"]["max"]),
            "alpha": random.uniform(loss["alpha"]["min"], loss["alpha"]["max"]),
        }

        reg = config["regularization"]
        params["regularization"] = {
            "l1_enabled": random.choice(reg["l1"]["enabled"]),
            "l1_strength": random.uniform(
                reg["l1"]["strength"]["min"], reg["l1"]["strength"]["max"]
            ),
            "l2_enabled": random.choice(reg["l2"]["enabled"]),
            "l2_strength": random.uniform(
                reg["l2"]["strength"]["min"], reg["l2"]["strength"]["max"]
            ),
            "dropout_enabled": random.choice(reg["dropout"]["enabled"]),
            "dropout_rate": random.uniform(
                reg["dropout"]["rate"]["min"], reg["dropout"]["rate"]["max"]
            ),
            "dropout_position": random.choice(reg["dropout"]["position"]),
            "dropconnect_enabled": random.choice(reg["dropconnect"]["enabled"]),
            "dropconnect_rate": random.uniform(
                reg["dropconnect"]["rate"]["min"], reg["dropconnect"]["rate"]["max"]
            ),
        }

        aug = config["augmentation"]
        params["augmentation"] = {
            "brightness_enabled": random.choice(aug["brightness"]["enabled"]),
            "brightness": random.uniform(
                aug["brightness"]["min"], aug["brightness"]["max"]
            ),
            "contrast_enabled": random.choice(aug["contrast"]["enabled"]),
            "contrast": random.uniform(aug["contrast"]["min"], aug["contrast"]["max"]),
            "flip_enabled": random.choice(aug["flip"]["enabled"]),
            "flip_prob": random.uniform(
                aug["flip"]["prob"]["min"], aug["flip"]["prob"]["max"]
            ),
            "crop_enabled": random.choice(aug["crop"]["enabled"]),
            "crop_size": random.randint(
                aug["crop"]["size"]["min"], aug["crop"]["size"]["max"]
            ),
            "upscale_enabled": random.choice(aug["upscale"]["enabled"]),
            "upscale": random.uniform(aug["upscale"]["min"], aug["upscale"]["max"]),
            "upscale_mode": random.choice(aug["upscale"]["mode"]),
        }

        return params

    @staticmethod
    def suggest_params(trial: optuna.Trial, config: Dict) -> Dict[str, Any]:
        params = {}

        arch = config["architecture"]
        params.update(
            {
                "conv_layers": trial.suggest_categorical(
                    "conv_layers", arch["conv_layers"]
                ),
                "kernel_size": trial.suggest_categorical(
                    "kernel_size", arch["kernel_size"]
                ),
                "stride": trial.suggest_categorical("stride", arch["stride"]),
                "padding": trial.suggest_categorical("padding", arch["padding"]),
                "use_batch_norm": trial.suggest_categorical(
                    "use_batch_norm", arch["use_batch_norm"]
                ),
                "use_group_norm": trial.suggest_categorical(
                    "use_group_norm", arch["use_group_norm"]
                ),
                "use_instance_norm": trial.suggest_categorical(
                    "use_instance_norm", arch["use_instance_norm"]
                ),
                "use_min_max_scaling": trial.suggest_categorical(
                    "use_min_max_scaling", arch["use_min_max_scaling"]
                ),
                "activation_conv": trial.suggest_categorical(
                    "activation_conv", arch["activation_conv"]
                ),
                "weight_initialization": trial.suggest_categorical(
                    "weight_initialization", arch["weight_initialization"]
                ),
                "freeze_layers": trial.suggest_categorical(
                    "freeze_layers", arch["freeze_layers"]
                ),
                "fully_connected_layers": trial.suggest_categorical(
                    "fully_connected_layers", arch["fully_connected_layers"]
                ),
                "activation_fc": trial.suggest_categorical(
                    "activation_fc", arch["activation_fc"]
                ),
            }
        )

        train = config["training"]
        params.update(
            {
                "gradient_clipping": trial.suggest_float(
                    "gradient_clipping",
                    train["gradient_clipping"]["min"],
                    train["gradient_clipping"]["max"],
                ),
                "early_stopping_enabled": trial.suggest_categorical(
                    "early_stopping_enabled", train["early_stopping"]["enabled"]
                ),
                "early_stopping_patience": trial.suggest_categorical(
                    "early_stopping_patience", train["early_stopping"]["patience"]
                ),
            }
        )

        opt = train["optimizer"]
        params.update(
            {
                "optimizer": trial.suggest_categorical("optimizer", opt["type"]),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    opt["learning_rate"]["min"],
                    opt["learning_rate"]["max"],
                    log=True,
                ),
                "momentum": trial.suggest_float(
                    "momentum", opt["momentum"]["min"], opt["momentum"]["max"]
                ),
                "weight_decay": trial.suggest_float(
                    "weight_decay",
                    opt["weight_decay"]["min"],
                    opt["weight_decay"]["max"],
                    log=True,
                ),
                "weight_decay_on_bias": trial.suggest_categorical(
                    "weight_decay_on_bias", opt["weight_decay_on_bias"]
                ),
            }
        )

        scheduler = train["scheduler"]
        params.update(
            {
                "scheduler": trial.suggest_categorical("scheduler", scheduler["type"]),
                "lr_decay_factor": trial.suggest_float(
                    "lr_decay_factor",
                    scheduler["lr_decay_factor"]["min"],
                    scheduler["lr_decay_factor"]["max"],
                ),
                "step_size": trial.suggest_categorical(
                    "step_size", scheduler["step_size"]
                ),
            }
        )

        loss = train["loss"]
        params.update(
            {
                "type": trial.suggest_categorical("loss_type", loss["types"]),
                "gamma": trial.suggest_float(
                    "loss_gamma", loss["gamma"]["min"], loss["gamma"]["max"]
                ),
                "alpha": trial.suggest_float(
                    "loss_alpha", loss["alpha"]["min"], loss["alpha"]["max"]
                ),
            }
        )

        reg = config["regularization"]
        params.update(
            {
                "l1_enabled": trial.suggest_categorical(
                    "l1_enabled", reg["l1"]["enabled"]
                ),
                "l1_strength": trial.suggest_float(
                    "l1_strength",
                    reg["l1"]["strength"]["min"],
                    reg["l1"]["strength"]["max"],
                    log=True,
                ),
                "l2_enabled": trial.suggest_categorical(
                    "l2_enabled", reg["l2"]["enabled"]
                ),
                "l2_strength": trial.suggest_float(
                    "l2_strength",
                    reg["l2"]["strength"]["min"],
                    reg["l2"]["strength"]["max"],
                    log=True,
                ),
                "dropout_enabled": trial.suggest_categorical(
                    "dropout_enabled", reg["dropout"]["enabled"]
                ),
                "dropout_rate": trial.suggest_float(
                    "dropout_rate",
                    reg["dropout"]["rate"]["min"],
                    reg["dropout"]["rate"]["max"],
                ),
                "dropout_position": trial.suggest_categorical(
                    "dropout_position", reg["dropout"]["position"]
                ),
                "dropconnect_enabled": trial.suggest_categorical(
                    "dropconnect_enabled", reg["dropconnect"]["enabled"]
                ),
                "dropconnect_rate": trial.suggest_float(
                    "dropconnect_rate",
                    reg["dropconnect"]["rate"]["min"],
                    reg["dropconnect"]["rate"]["max"],
                ),
            }
        )

        aug = config["augmentation"]

        params.update(
            {
                "brightness_enabled": trial.suggest_categorical(
                    "brightness_enabled", aug["brightness"]["enabled"]
                ),
                "brightness": trial.suggest_float(
                    "brightness",
                    aug["brightness"]["min"],
                    aug["brightness"]["max"],
                ),
                "contrast_enabled": trial.suggest_categorical(
                    "contrast_enabled", aug["contrast"]["enabled"]
                ),
                "contrast": trial.suggest_float(
                    "contrast", aug["contrast"]["min"], aug["contrast"]["max"]
                ),
                "flip_enabled": trial.suggest_categorical(
                    "flip_enabled", aug["flip"]["enabled"]
                ),
                "flip_prob": trial.suggest_float(
                    "flip_prob", aug["flip"]["prob"]["min"], aug["flip"]["prob"]["max"]
                ),
                "crop_enabled": trial.suggest_categorical(
                    "crop_enabled", aug["crop"]["enabled"]
                ),
                "crop_size": trial.suggest_int(
                    "crop_size", aug["crop"]["size"]["min"], aug["crop"]["size"]["max"]
                ),
                "upscale_enabled": trial.suggest_categorical(
                    "upscale_enabled", aug["upscale"]["enabled"]
                ),
                "upscale": trial.suggest_float(
                    "upscale", aug["upscale"]["min"], aug["upscale"]["max"]
                ),
                "upscale_mode": trial.suggest_categorical(
                    "upscale_mode", aug["upscale"]["mode"]
                ),
            }
        )

        return params
