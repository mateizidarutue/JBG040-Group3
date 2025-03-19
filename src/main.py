import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from src.trainer.trainer import Trainer
from src.search.optuna_bayes_hyperband import OptunaBayesHyperband
from src.dataset.image_dataset import ImageDataset
import os


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
        torch.tensor(full_train_dataset.targets[train_dataset.indices])
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
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def main():
    search_config_path = "src/config/search_config.yaml"
    static_config_path = "src/config/static_config.yaml"
    search_config = load_config(search_config_path)
    static_config = load_config(static_config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = setup_dataloaders(
        x_train_path=Path(static_config["train_data"]),
        y_train_path=Path(static_config["train_labels"]),
        x_test_path=Path(static_config["test_data"]),
        y_test_path=Path(static_config["test_labels"]),
        batch_size=static_config["batch_size"],
    )

    trainer = Trainer(
        config=static_config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    search = OptunaBayesHyperband(
        min_budget=static_config["min_budget"],
        max_budget=static_config["max_epochs"],
        eta=static_config["eta"],
        trials_per_batch=static_config["trials_per_batch"],
        number_of_runs=static_config["number_of_runs"],
        training_fn=trainer.train,
        config=search_config,
        study_name="cnn_hyperparameter_search",
        direction=static_config["direction"],
    )

    search.run()


if __name__ == "__main__":
    main()
