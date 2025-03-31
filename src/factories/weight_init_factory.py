import torch.nn as nn
from typing import Callable


class WeightInitFactory:
    @staticmethod
    def get_initializer(method: str) -> Callable[[nn.Module], None]:
        def init_fn(module: nn.Module):
            WeightInitFactory._init_weight(module, method)

        return init_fn

    @staticmethod
    def _init_weight(module: nn.Module, init_method: str):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if init_method == "xavier":
                nn.init.xavier_uniform_(module.weight)
            elif init_method == "kaiming":
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            elif init_method == "normal":
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif init_method == "uniform":
                nn.init.uniform_(module.weight, a=-0.1, b=0.1)
            else:
                raise ValueError(
                    f"Unsupported weight initialization method: {init_method}"
                )

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
