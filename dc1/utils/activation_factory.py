import torch.nn as nn
from typing import Dict, Any

class ActivationFactory:
    @staticmethod
    def get_activation(config: Dict[str, Any]) -> nn.Module:
        name = config["activation"]

        if name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'leaky_relu':
            return nn.LeakyReLU(inplace=True)
        elif name == 'prelu':
            return nn.PReLU()
        elif name == 'gelu':
            return nn.GELU()
        elif name == 'swish':
            return nn.SiLU()
        elif name == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {name}")