from pathlib import Path
import json
import yaml
import torch

from src.dataset.data_loader_manager import DataLoaderManager
from src.trainer.trainer import Trainer



def train_single_config(
    params, static_config, train_loader, val_loader, test_loader, device, run_id
):
    trainer = Trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        input_size=static_config["input_size"],
        num_classes=static_config["num_classes"],
    )

    print(f"\nStarting training for configuration {run_id}...")
    trainer.train(params, static_config["max_budget"])

    print(f"Run {run_id} completed!")


def main():
    with open("src/config/static_config.yaml", "r") as f:
        static_config = yaml.safe_load(f)

    with open("src/config/model_config.json", "r") as f:
        configs = json.load(f)

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = DataLoaderManager.setup_dataloaders(
        Path(static_config["train_data"]),
        Path(static_config["train_labels"]),
        Path(static_config["test_data"]),
        Path(static_config["test_labels"]),
        static_config["batch_size"],
    )

    print(f"Found {len(configs)} configurations to train")

    for i, params in enumerate(configs):
        run_id = i + 1
        train_single_config(
            params, static_config, train_loader, val_loader, test_loader, device, run_id
        )

    print("\nAll training runs completed!")
    print


if __name__ == "__main__":
    main()
