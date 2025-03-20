import json
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.models.cnn import CNN
from src.trainer.trainer import Trainer
from src.dataset.image_dataset import ImageDataset
import os


def setup_dataloaders(
    x_train_path: Path,
    y_train_path: Path,
    x_val_path: Path,
    y_val_path: Path,
    batch_size: int,
):
    train_dataset = ImageDataset(x_train_path, y_train_path)
    val_dataset = ImageDataset(x_val_path, y_val_path)

    class_counts = torch.bincount(torch.LongTensor(train_dataset.targets))
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[train_dataset.targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # Disable multiprocessing to avoid pickling issues
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0,  # Set to 0 to disable multiprocessing
        persistent_workers=False,  # Disable persistent workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 to disable multiprocessing
        persistent_workers=False,  # Disable persistent workers
    )

    return train_loader, val_loader


def train_single_config(
    params, static_config, train_loader, val_loader, device, run_id
):
    trainer = Trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None,
        device=str(device),
        input_size=static_config["input_size"],
        num_classes=static_config["num_classes"],
    )

    print(f"\nStarting training for configuration {run_id}...")
    _, history, model = trainer.train_model(params, static_config["max_budget"])
    model: CNN
    val_loss, metrics = trainer.test(model, trainer.val_loader, params, True)

    output_dir = Path(f"model_output/run_{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.pt"
    result_path = output_dir / "result.json"

    torch.save(model.state_dict(), model_path)

    with open(result_path, "w") as f:
        json.dump(
            {
                "parameters": params,
                "test_loss": val_loss,
                "metrics": metrics,
                "history": history,
            },
            f,
            indent=4,
        )

    print(f"Run {run_id} completed!")
    print(f"Model saved to: {model_path}")
    print(f"Result saved to: {result_path}")

    return val_loss, metrics


def main():
    with open("src/config/static_config.yaml", "r") as f:
        static_config = yaml.safe_load(f)

    with open("src/config/model_config.json", "r") as f:
        configs = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = setup_dataloaders(
        Path(static_config["train_data"]),
        Path(static_config["train_labels"]),
        Path(static_config["test_data"]),
        Path(static_config["test_labels"]),
        static_config["batch_size"],
    )

    best_loss = float("inf")
    best_config = None
    best_metrics = None
    best_run_id = None

    print(f"Found {len(configs)} configurations to train")

    for i, params in enumerate(configs):
        run_id = i + 1
        val_loss, metrics = train_single_config(
            params, static_config, train_loader, val_loader, device, run_id
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_config = params
            best_metrics = metrics
            best_run_id = run_id

    print("\nAll training runs completed!")
    print(f"Best performing model was from run {best_run_id}")
    print(f"Best validation loss: {best_loss:.4f}")
    print("Best metrics:", best_metrics)

    summary_path = Path("model_output/best_run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "best_run_id": best_run_id,
                "best_loss": best_loss,
                "best_config": best_config,
                "best_metrics": best_metrics,
            },
            f,
            indent=4,
        )
    print(f"Best run summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
