from typing import Dict, Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from dc1.image_dataset import ImageDataset


class LossFactory:
    """Factory class for managing loss functions specialized for lung disease classification.
    Class labels:
    0: Atelectasis
    1: Effusion
    2: Infiltration
    3: No finding
    4: Nodule
    5: Pneumothorax
    """

    def __init__(self, dataset: ImageDataset, device: str = "cpu") -> None:
        """Initialize loss functions with class distribution from dataset.

        Args:
            dataset: Training dataset to calculate class distribution from
            device: Device to place tensors on ("cpu" or "cuda")

        Raises:
            ValueError: If any class has zero samples
        """

        # Calculate class distribution
        class_counts = torch.bincount(torch.tensor(dataset.targets, device=device), minlength=6)

        # Validate class distribution
        if torch.all(class_counts == 0):
            raise ValueError(
                "No samples found in training data. Cannot calculate class weights."
            )

        zero_classes = torch.where(class_counts == 0)[0].tolist()
        if zero_classes:
            class_names = [
                "Atelectasis",
                "Effusion",
                "Infiltration",
                "No finding",
                "Nodule",
                "Pneumothorax",
            ]
            zero_class_names = [class_names[i] for i in zero_classes]
            raise ValueError(
                f"Found classes with zero samples: {zero_class_names}. "
                "All classes must have at least one sample to calculate weights."
            )

        # Calculate weights once during initialization
        self.class_weights = self.calculate_class_weights(class_counts)

        self._loss_functions: Dict[str, Callable[..., torch.Tensor]] = {
            "ce": nn.CrossEntropyLoss(),
            "recall_weighted_ce": self._get_recall_weighted_ce(),
            "hierarchical_ce": self._get_hierarchical_ce(),
            "focal": self._get_focal_loss(),
            "dice": self._get_dice_loss(),
            "tversky": self._get_tversky_loss(),
            "combined": self._get_combined_loss(),
        }

    def calculate_class_weights(self, class_counts: torch.Tensor) -> torch.Tensor:
        """Calculate weights based on class distribution and clinical importance.

        Args:
            class_counts: Tensor containing count of samples for each class

        Returns:
            Tensor of weights for each class
        """
        # Inverse frequency weighting
        total_samples = class_counts.sum()
        inverse_freqs = total_samples / (class_counts * len(class_counts))

        # Clinical importance multiplier (higher for critical conditions)
        # Pneumothorax (5) and Nodule (4) are most critical
        # No finding (3) gets lower weight
        clinical_importance = torch.tensor([1.5, 1.5, 1.5, 0.5, 2.0, 2.0])

        # Combine frequency-based weights with clinical importance
        weights = inverse_freqs * clinical_importance

        # Normalize weights to prevent extreme values
        weights = weights / weights.mean()
        return weights

    def _get_recall_weighted_ce(self) -> nn.CrossEntropyLoss:
        """Creates recall-focused weighted cross-entropy loss using calculated weights."""
        return nn.CrossEntropyLoss(weight=self.class_weights)

    def _get_focal_loss(
        self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None
    ) -> Callable[..., torch.Tensor]:
        """Creates focal loss that reduces impact of dominant 'No Finding' cases.

        Args:
            gamma: Focusing parameter that reduces the loss contribution from easy examples
            alpha: Optional class weights tensor
        """
        if alpha is None:
            # Use calculated weights for alpha
            alpha = self.class_weights / self.class_weights.sum()

        def focal_loss(
            predictions: torch.Tensor, targets: torch.Tensor
        ) -> torch.Tensor:
            ce_loss = F.cross_entropy(predictions, targets, reduction="none")
            pt = torch.exp(-ce_loss)

            # Get alpha weight for each target
            alpha_t = alpha[targets]

            # Calculate focal loss with alpha weighting
            focal_weight = alpha_t * (1 - pt) ** gamma

            # Higher weight for disease classes, lower for 'No finding'
            is_no_finding = (targets == 3).float()
            class_weight = self.class_weights[targets]

            return (focal_weight * ce_loss * class_weight).mean()

        return focal_loss

    def _get_dice_loss(self) -> Callable[..., torch.Tensor]:
        """Creates Dice loss for better recall on rare diseases and small lesions."""

        def dice_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            # Convert predictions to probabilities
            probs = F.softmax(predictions, dim=1)

            # Convert targets to one-hot
            targets_one_hot = F.one_hot(targets, num_classes=6).float()

            # Calculate Dice coefficient for each class
            numerator = 2.0 * (probs * targets_one_hot).sum(dim=0)
            denominator = probs.sum(dim=0) + targets_one_hot.sum(dim=0)

            # Add small epsilon to avoid division by zero
            dice_coef = numerator / (denominator + 1e-7)

            # Use calculated weights
            weighted_dice = dice_coef * self.class_weights

            # Return negative mean (since we minimize loss)
            return 1.0 - weighted_dice.mean()

        return dice_loss

    def _get_tversky_loss(self, beta: float = 0.7) -> Callable[..., torch.Tensor]:
        """Creates Tversky loss with beta > 0.5 to penalize false negatives more than false positives.

        Args:
            beta: Weight of false negatives (> 0.5 to penalize FN more than FP)
        """

        def tversky_loss(
            predictions: torch.Tensor, targets: torch.Tensor
        ) -> torch.Tensor:
            # Convert predictions to probabilities
            probs = F.softmax(predictions, dim=1)
            targets_one_hot = F.one_hot(targets, num_classes=6).float()

            # Calculate true positives, false positives, and false negatives
            tp = (probs * targets_one_hot).sum(dim=0)
            fp = (probs * (1 - targets_one_hot)).sum(dim=0)
            fn = ((1 - probs) * targets_one_hot).sum(dim=0)

            # Tversky index with beta > 0.5 to focus more on false negatives
            numerator = tp
            denominator = tp + beta * fn + (1 - beta) * fp + 1e-7

            # Use calculated weights
            tversky_index = (numerator / denominator) * self.class_weights

            return 1.0 - tversky_index.mean()

        return tversky_loss

    def _get_hierarchical_ce(self) -> Callable[..., torch.Tensor]:
        """Creates hierarchical cross-entropy with class weighting."""

        def hierarchical_loss(
            predictions: torch.Tensor, targets: torch.Tensor
        ) -> torch.Tensor:
            # First level: Disease vs No Disease
            no_finding_mask = targets == 3  # Class 3 is 'No finding'
            binary_targets = (
                ~no_finding_mask
            ).float()  # 1 for disease, 0 for no finding

            # Convert predictions to binary disease vs no disease
            binary_predictions = predictions.clone()
            binary_predictions[:, 3] = predictions[:, 3]  # No finding score
            disease_scores = torch.cat(
                [predictions[:, :3], predictions[:, 4:]],  # Classes 0-2  # Classes 4-5
                dim=1,
            ).sum(dim=1)
            binary_predictions = torch.stack(
                [disease_scores, binary_predictions[:, 3]], dim=1
            )

            # Calculate binary weights from class weights
            no_finding_weight = self.class_weights[3]
            disease_weight = self.class_weights[
                self.class_weights != self.class_weights[3]
            ].mean()
            binary_weights = torch.tensor([disease_weight, no_finding_weight])

            # First level loss (disease vs no disease) with weights
            level1_loss = F.cross_entropy(
                binary_predictions, binary_targets.long(), weight=binary_weights
            )

            # Second level loss (between diseases) - only for disease cases
            disease_cases = (~no_finding_mask).nonzero().squeeze(1)
            if len(disease_cases) > 0:
                disease_predictions = torch.cat(
                    [
                        predictions[disease_cases, :3],  # Classes 0-2
                        predictions[disease_cases, 4:],  # Classes 4-5
                    ],
                    dim=1,
                )

                # Map target indices for disease-only classification
                disease_targets = targets[disease_cases].clone()
                disease_targets[
                    disease_targets > 3
                ] -= 1  # Adjust indices after removing 'No finding'

                # Use calculated weights for diseases
                disease_weights = torch.cat(
                    [self.class_weights[:3], self.class_weights[4:]]
                )
                level2_loss = F.cross_entropy(
                    disease_predictions, disease_targets, weight=disease_weights
                )
            else:
                level2_loss = torch.tensor(0.0, device=predictions.device)

            # Combine losses with higher weight on disease detection
            return 0.6 * level1_loss + 0.4 * level2_loss

        return hierarchical_loss

    def _get_combined_loss(self) -> Callable[..., torch.Tensor]:
        """Creates a combined loss using Focal, Dice, and Tversky losses."""
        focal = self._get_focal_loss()
        dice = self._get_dice_loss()
        tversky = self._get_tversky_loss()

        def combined_loss(
            predictions: torch.Tensor, targets: torch.Tensor
        ) -> torch.Tensor:
            # Combine losses with weights prioritizing false negative reduction
            focal_weight = 0.4  # Good for handling class imbalance
            dice_weight = 0.3  # Good for small lesions
            tversky_weight = 0.3  # Extra focus on false negatives

            return (
                focal_weight * focal(predictions, targets)
                + dice_weight * dice(predictions, targets)
                + tversky_weight * tversky(predictions, targets)
            )

        return combined_loss

    def get_loss(self, name: str) -> Callable[..., torch.Tensor]:
        if name not in self._loss_functions:
            raise ValueError(
                f"Unknown loss function: {name}. Available options: {list(self._loss_functions.keys())}"
            )
        return self._loss_functions[name]

    def add_custom_loss(self, name: str, loss_fn: Callable[..., torch.Tensor]) -> None:
        """Add a custom loss function."""
        self._loss_functions[name] = loss_fn
