import torch.nn as nn
from typing import Dict, Any


class ActivationFactory:
    @staticmethod
    def get_activation(activation_type: str) -> nn.Module:
        if activation_type == "relu":
            return nn.ReLU(inplace=True)
        elif activation_type == "leaky_relu":
            return nn.LeakyReLU(inplace=True)
        elif activation_type == "prelu":
            return nn.PReLU()
        elif activation_type == "gelu":
            return nn.GELU()
        elif activation_type == "swish":
            return nn.SiLU()
        elif activation_type == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation_type}")
