import torch.optim as optim
import torch.nn as nn
from typing import Dict, Any


class OptimizerFactory:
    @staticmethod
    def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
        optimizer_type = config["optimizer"]
        learning_rate = config["learning_rate"]
        momentum = config["momentum"]
        weight_decay = config["weight_decay"]
        weight_decay_on_bias = config["weight_decay_on_bias"]

        if not weight_decay_on_bias:
            params = [
                {
                    "params": [
                        p for n, p in model.named_parameters() if not n.endswith("bias")
                    ],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [
                        p for n, p in model.named_parameters() if n.endswith("bias")
                    ],
                    "weight_decay": 0.0,
                },
            ]
        else:
            params = model.parameters()

        if optimizer_type == "adam":
            return optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == "sgd":
            return optim.SGD(
                params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay
            )
        elif optimizer_type == "rmsprop":
            return optim.RMSprop(
                params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
