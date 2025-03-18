import numpy as np
from typing import Dict
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(
    outputs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> Dict[str, float]:
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    if outputs.shape[1] > 1:
        predictions = np.argmax(outputs, axis=1)
    else:
        predictions = (outputs > threshold).astype(int)

    accuracy = accuracy_score(targets, predictions)

    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        targets, predictions, average="macro", zero_division=0
    )

    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        targets, predictions, average="micro", zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
    }

    for i in range(len(precision)):
        metrics.update(
            {
                f"class_{i}_precision": float(precision[i]),
                f"class_{i}_recall": float(recall[i]),
                f"class_{i}_f1": float(f1[i]),
                f"class_{i}_support": int(support[i]),
            }
        )

    return metrics
