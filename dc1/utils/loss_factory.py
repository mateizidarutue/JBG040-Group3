from utils.losses import FocalLoss, DiceLoss, TverskyLoss, CombinedLoss
from typing import Dict, Any
import torch.nn as nn

class LossFactory:
    @staticmethod
    def get_loss(config: Dict[str, Any]) -> nn.Module:
        loss_type = config['loss_function']['type']
        gamma = config['loss_function']['gamma']
        alpha = config['loss_function']['alpha']

        if loss_type == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif loss_type == 'focal':
            return FocalLoss(gamma=gamma, alpha=alpha)
        elif loss_type == 'dice':
            return DiceLoss()
        elif loss_type == 'tversky':
            return TverskyLoss(alpha=alpha, beta=(1 - alpha))
        elif loss_type == 'combined':
            return CombinedLoss(gamma=gamma, alpha=alpha, use_tversky=True)
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")
