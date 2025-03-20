from typing import Dict, Any
import optuna


class ParamSampler:
    @staticmethod
    def suggest_params(trial: optuna.Trial, config: Dict[str, Any]) -> Dict[str, Any]:
        params = {}

        arch = config["architecture"]
        conv_layers_str: str = trial.suggest_categorical(
            "conv_layers", arch["conv_layers"]
        )
        params["conv_layers"] = [int(x) for x in conv_layers_str.split(",")]

        params["kernel_size"] = trial.suggest_categorical(
            "kernel_size", arch["kernel_size"]
        )
        params["stride"] = trial.suggest_categorical("stride", arch["stride"])
        params["use_batch_norm"] = trial.suggest_categorical(
            "use_batch_norm", arch["use_batch_norm"]
        )
        params["use_group_norm"] = trial.suggest_categorical(
            "use_group_norm", arch["use_group_norm"]
        )
        params["use_instance_norm"] = trial.suggest_categorical(
            "use_instance_norm", arch["use_instance_norm"]
        )
        params["use_min_max_scaling"] = trial.suggest_categorical(
            "use_min_max_scaling", arch["use_min_max_scaling"]
        )
        params["activation_conv"] = trial.suggest_categorical(
            "activation_conv", arch["activation_conv"]
        )
        params["weight_initialization"] = trial.suggest_categorical(
            "weight_initialization", arch["weight_initialization"]
        )
        fully_connected_layers_str: str = trial.suggest_categorical(
            "fully_connected_layers", arch["fully_connected_layers"]
        )
        params["fully_connected_layers"] = [
            int(x) for x in fully_connected_layers_str.split(",")
        ]
        params["activation_fc"] = trial.suggest_categorical(
            "activation_fc", arch["activation_fc"]
        )

        train = config["training"]

        grad_clip = train["gradient_clipping"]
        params["gradient_clipping_enabled"] = trial.suggest_categorical(
            "gradient_clipping_enabled", grad_clip["enabled"]
        )
        if params["gradient_clipping_enabled"]:
            params["gradient_clipping"] = trial.suggest_float(
                "gradient_clipping", grad_clip["min"], grad_clip["max"], log=True
            )
        else:
            params["gradient_clipping"] = None

        opt_conf = train["optimizer"]
        optimizer_choice = trial.suggest_categorical("optimizer", opt_conf["type"])
        params["optimizer"] = optimizer_choice
        params["learning_rate"] = trial.suggest_float(
            "learning_rate",
            opt_conf["learning_rate"]["min"],
            opt_conf["learning_rate"]["max"],
            log=True,
        )
        if optimizer_choice in ["sgd", "rmsprop"]:
            params["momentum"] = trial.suggest_float(
                "momentum", opt_conf["momentum"]["min"], opt_conf["momentum"]["max"]
            )
        else:
            params["momentum"] = None
        params["weight_decay"] = trial.suggest_float(
            "weight_decay",
            opt_conf["weight_decay"]["min"],
            opt_conf["weight_decay"]["max"],
            log=True,
        )
        params["weight_decay_on_bias"] = trial.suggest_categorical(
            "weight_decay_on_bias", opt_conf["weight_decay_on_bias"]
        )

        sch_conf = train["scheduler"]
        scheduler_choice = trial.suggest_categorical("scheduler", sch_conf["type"])
        params["scheduler"] = scheduler_choice
        params["lr_decay_factor"] = trial.suggest_float(
            "lr_decay_factor",
            sch_conf["lr_decay_factor"]["min"],
            sch_conf["lr_decay_factor"]["max"],
        )
        if scheduler_choice == "step":
            params["step_size"] = trial.suggest_categorical(
                "step_size", sch_conf["step_size"]
            )
        else:
            params["step_size"] = None

        loss_conf = train["loss_function"]
        loss_type = trial.suggest_categorical("loss_type", loss_conf["type"])
        params["loss_type"] = loss_type
        if loss_type in ["focal", "tversky", "combined"]:
            if loss_type in ["focal", "combined"]:
                params["loss_gamma"] = trial.suggest_float(
                    "loss_gamma", loss_conf["gamma"]["min"], loss_conf["gamma"]["max"]
                )
            else:
                params["loss_gamma"] = None

            params["loss_alpha"] = trial.suggest_float(
                "loss_alpha", loss_conf["alpha"]["min"], loss_conf["alpha"]["max"]
            )
        else:
            params["loss_gamma"] = None
            params["loss_alpha"] = None

        reg_conf = config["regularization"]
        params["l1_enabled"] = trial.suggest_categorical(
            "l1_enabled", reg_conf["l1"]["enabled"]
        )
        if params["l1_enabled"]:
            params["l1_strength"] = trial.suggest_float(
                "l1_strength",
                reg_conf["l1"]["strength"]["min"],
                reg_conf["l1"]["strength"]["max"],
                log=True,
            )
        else:
            params["l1_strength"] = None

        params["l2_enabled"] = trial.suggest_categorical(
            "l2_enabled", reg_conf["l2"]["enabled"]
        )
        if params["l2_enabled"]:
            params["l2_strength"] = trial.suggest_float(
                "l2_strength",
                reg_conf["l2"]["strength"]["min"],
                reg_conf["l2"]["strength"]["max"],
                log=True,
            )
        else:
            params["l2_strength"] = None

        params["dropout_enabled"] = trial.suggest_categorical(
            "dropout_enabled", reg_conf["dropout"]["enabled"]
        )
        if params["dropout_enabled"]:
            params["dropout_rate"] = trial.suggest_float(
                "dropout_rate",
                reg_conf["dropout"]["rate"]["min"],
                reg_conf["dropout"]["rate"]["max"],
            )
            params["dropout_position"] = trial.suggest_categorical(
                "dropout_position", reg_conf["dropout"]["position"]
            )
        else:
            params["dropout_rate"] = None
            params["dropout_position"] = None

        aug_conf = config["augmentation"]
        params["brightness_enabled"] = trial.suggest_categorical(
            "brightness_enabled", aug_conf["brightness"]["enabled"]
        )
        if params["brightness_enabled"]:
            params["brightness"] = trial.suggest_float(
                "brightness",
                aug_conf["brightness"]["min"],
                aug_conf["brightness"]["max"],
            )
        else:
            params["brightness"] = None

        params["contrast_enabled"] = trial.suggest_categorical(
            "contrast_enabled", aug_conf["contrast"]["enabled"]
        )
        if params["contrast_enabled"]:
            params["contrast"] = trial.suggest_float(
                "contrast", aug_conf["contrast"]["min"], aug_conf["contrast"]["max"]
            )
        else:
            params["contrast"] = None

        params["rotation_enabled"] = trial.suggest_categorical(
            "rotation_enabled", aug_conf["rotation"]["enabled"]
        )

        params["crop_enabled"] = trial.suggest_categorical(
            "crop_enabled", aug_conf["crop"]["enabled"]
        )
        if params["crop_enabled"]:
            params["crop_size"] = trial.suggest_int(
                "crop_size",
                aug_conf["crop"]["size"]["min"],
                aug_conf["crop"]["size"]["max"],
            )
        else:
            params["crop_size"] = None

        return params
