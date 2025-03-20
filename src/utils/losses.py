import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.loss = nn.CrossEntropyLoss()
        self.max_ce_loss = -torch.log(torch.tensor(1.0 / num_classes))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.loss(inputs, targets)
        return ce_loss / self.max_ce_loss


class FocalLoss(nn.Module):
    def __init__(self, gamma: float, alpha: float) -> None:
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.min_val = float("inf")
        self.max_val = float("-inf")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        self.min_val = min(self.min_val, focal_loss.mean().item())
        self.max_val = max(self.max_val, focal_loss.mean().item())

        if self.max_val > self.min_val:
            focal_loss = (focal_loss.mean() - self.min_val) / (
                self.max_val - self.min_val + 1e-8
            )
        else:
            focal_loss = focal_loss.mean()

        return focal_loss


class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 1 and inputs.size(1) > 1:
            targets = F.one_hot(targets, num_classes=inputs.size(1)).float()

        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth
        )
        return 1 - dice


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float, beta: float) -> None:
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = 1.0

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 1 and inputs.size(1) > 1:
            targets = F.one_hot(targets, num_classes=inputs.size(1)).float()

        inputs = torch.sigmoid(inputs)
        true_pos = (inputs * targets).sum()
        false_neg = ((1 - inputs) * targets).sum()
        false_pos = (inputs * (1 - targets)).sum()

        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth
        )
        return 1 - tversky


class CombinedLoss(nn.Module):
    def __init__(
        self, gamma: float, alpha: float, use_tversky: bool, num_classes: int
    ) -> None:
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)
        self.dice_loss = DiceLoss()
        self.tversky_loss = TverskyLoss(alpha=0.5, beta=0.5)
        self.cross_entropy = CrossEntropyLoss(num_classes)
        self.use_tversky = use_tversky

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        ce_loss = self.cross_entropy(inputs, targets)
        tversky = self.tversky_loss(inputs, targets)

        combined_loss = 0.25 * focal + 0.25 * dice + 0.25 * ce_loss + 0.25 * tversky
        return combined_loss
