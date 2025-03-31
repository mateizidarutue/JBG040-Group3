import torch.optim as optim
from typing import Dict, Any


class SchedulerFactory:
    @staticmethod
    def get_scheduler(
        optimizer: optim.Optimizer, params: Dict[str, Any]
    ) -> optim.lr_scheduler._LRScheduler:
        scheduler_type = params["scheduler"]
        decay_factor = params["lr_decay_factor"]
        step_size = params["step_size"]

        if scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=decay_factor
            )

        elif scheduler_type == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=decay_factor, patience=5
            )

        elif scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-5
            )

        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
