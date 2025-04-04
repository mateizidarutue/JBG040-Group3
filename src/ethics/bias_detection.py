from collections import defaultdict
import torch


def initialize_bias_monitor():
    return defaultdict(lambda: {
        "total": 0,
        "correct": 0,
        "predicted_as": defaultdict(int),
        "confidences": []
    })


def update_bias_monitor(monitor, true_class, predicted_class, logits):
    confidence = torch.softmax(logits, dim=1)[0, predicted_class].item()

    monitor[true_class]["total"] += 1
    monitor[true_class]["predicted_as"][predicted_class] += 1
    monitor[true_class]["confidences"].append(confidence)

    if predicted_class == true_class:
        monitor[true_class]["correct"] += 1


def generate_bias_report(monitor):
    print("\n Class-wise Bias Report:")
    for cls, stats in monitor.items():
        total = stats["total"]
        correct = stats["correct"]
        acc = correct / total if total > 0 else 0
        avg_conf = sum(stats["confidences"]) / len(stats["confidences"]) if stats["confidences"] else 0

        print(f"\n Class {cls}:")
        print(f"   Accuracy: {acc:.2%}")
        print(f"   Total samples: {total}")
        print(f"   Avg confidence: {avg_conf:.2f}")
        print("   Predicted as:")
        for pred_cls, count in stats["predicted_as"].items():
            print(f"     → {pred_cls}: {count}x")