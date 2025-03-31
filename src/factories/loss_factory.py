from typing import Dict, Any
import torch.nn as nn

from src.factories.losses import (
    FocalLoss,
    DiceLoss,
    TverskyLoss,
    CombinedLoss,
    CrossEntropyLoss,
)


class LossFactory:
    @staticmethod
    def get_loss(params: Dict[str, Any], num_classes: int) -> nn.Module:
        loss_type = params["loss_type"]
        gamma = params["loss_gamma"]
        alpha = params["loss_alpha"]

        if loss_type == "cross_entropy":
            return CrossEntropyLoss(num_classes=num_classes)
        elif loss_type == "focal":
            return FocalLoss(gamma=gamma, alpha=alpha)
        elif loss_type == "dice":
            return DiceLoss()
        elif loss_type == "tversky":
            return TverskyLoss(alpha=alpha, beta=(1 - alpha))
        elif loss_type == "combined":
            return CombinedLoss(
                gamma=gamma, alpha=alpha, use_tversky=True, num_classes=num_classes
            )
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")
