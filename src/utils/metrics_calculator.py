import torch
from sklearn.metrics import confusion_matrix

from src.types.evaluation_metrics import EvaluationMetrics


class MetricsCalculator:
    def compute(self, outputs: torch.Tensor, labels: torch.Tensor) -> EvaluationMetrics:
        if outputs.shape[1] > 1:
            predictions = torch.argmax(outputs, dim=1).numpy()
        else:
            predictions = (outputs > 0.5).int().numpy()

        labels = labels.numpy()

        conf_matrix = confusion_matrix(labels, predictions)
        num_classes = conf_matrix.shape[0]

        correct = conf_matrix.trace()
        total = conf_matrix.sum()
        accuracy = correct / total if total else 0.0

        precision = []
        recall = []
        f1 = []

        for i in range(num_classes):
            tp = conf_matrix[i, i]
            fp = conf_matrix[:, i].sum() - tp
            fn = conf_matrix[i, :].sum() - tp

            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f = 2 * p * r / (p + r) if (p + r) else 0.0

            precision.append(p)
            recall.append(r)
            f1.append(f)

        macro_precision = sum(precision) / num_classes
        macro_recall = sum(recall) / num_classes
        macro_f1 = sum(f1) / num_classes

        no_finding_idx = 3
        score = 0.0
        for true_class in range(num_classes):
            for pred_class in range(num_classes):
                count = conf_matrix[true_class, pred_class]
                if true_class == pred_class:
                    continue
                if true_class != no_finding_idx and pred_class == no_finding_idx:
                    score += count * 1.0
                elif true_class != no_finding_idx and pred_class != no_finding_idx:
                    score += count * 0.5
                elif true_class == no_finding_idx and pred_class != no_finding_idx:
                    score += count * 0.25

        return EvaluationMetrics(
            accuracy=accuracy,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            macro_f1=macro_f1,
            class_precision={i: precision[i] for i in range(num_classes)},
            class_recall={i: recall[i] for i in range(num_classes)},
            class_f1={i: f1[i] for i in range(num_classes)},
            score=score,
        )
