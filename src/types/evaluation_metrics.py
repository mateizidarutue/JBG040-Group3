from dataclasses import dataclass
from typing import Dict


@dataclass
class EvaluationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    class_precision: Dict[int, float]
    class_recall: Dict[int, float]
    class_f1: Dict[int, float]
    score: float

    def to_dict(self):
        base = {
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "score": self.score,
        }

        for i in self.class_precision:
            base[f"class_{i}_precision"] = self.class_precision[i]
            base[f"class_{i}_recall"] = self.class_recall[i]
            base[f"class_{i}_f1"] = self.class_f1[i]

        return base
