import torch
from typing import Dict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class MetricsCalculator:
    def compute(
        self, outputs: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5
    ) -> Dict[str, float]:
        if outputs.shape[1] > 1:
            predictions = torch.argmax(outputs, dim=1).numpy()
        else:
            predictions = (outputs > threshold).int().numpy()

        labels = labels.numpy()

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="macro", zero_division=0
        )

        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )

        metrics = {
            "accuracy": accuracy,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1,
        }

        for i in range(len(class_precision)):
            metrics.update(
                {
                    f"class_{i}_precision": class_precision[i],
                    f"class_{i}_recall": class_recall[i],
                    f"class_{i}_f1": class_f1[i],
                }
            )

        return metrics
