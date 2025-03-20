import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from src.trainer.trainer import Trainer
from src.search.optuna_bayes_hyperband import OptunaBayesHyperband
from src.dataset.image_dataset import ImageDataset
import os
import optuna
import logging
import sys


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_dataloaders(
    x_train_path: Path,
    y_train_path: Path,
    x_test_path: Path,
    y_test_path: Path,
    batch_size: int,
):
    full_train_dataset = ImageDataset(x_train_path, y_train_path)
    test_dataset = ImageDataset(x_test_path, y_test_path)

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    class_counts = torch.bincount(
        torch.LongTensor(full_train_dataset.targets[train_dataset.indices])
    )

    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[full_train_dataset.targets[train_dataset.indices]]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    num_workers = os.cpu_count() // 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader


def main():
    search_config_path = "src/config/search_config.yaml"
    static_config_path = "src/config/static_config.yaml"
    search_config = load_config(search_config_path)
    static_config = load_config(static_config_path)

    optuna.logging.set_verbosity(optuna.logging.INFO)
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = setup_dataloaders(
        x_train_path=Path(static_config["train_data"]),
        y_train_path=Path(static_config["train_labels"]),
        x_test_path=Path(static_config["test_data"]),
        y_test_path=Path(static_config["test_labels"]),
        batch_size=static_config["batch_size"],
    )

    trainer = Trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        input_size=static_config["input_size"],
        num_classes=static_config["num_classes"],
    )

    search = OptunaBayesHyperband(
        min_budget=static_config["min_budget"],
        max_budget=static_config["max_budget"],
        eta=static_config["eta"],
        total_trials=static_config["total_trials"],
        num_classes=static_config["num_classes"],
        trainer=trainer,
        test_loader=test_loader,
        config=search_config,
        study_name="cnn_hyperparameter_search",
        direction=static_config["direction"],
    )

    search.run()


if __name__ == "__main__":
    main()
