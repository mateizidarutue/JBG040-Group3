from pathlib import Path
import torch
import yaml

from src.trainer.trainer import Trainer
from src.search.optuna_bayes_hyperband import OptunaBayesHyperband
from src.dataset.data_loader_manager import DataLoaderManager


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    search_config_path = Path("src/config/search_config.yaml")
    static_config_path = Path("src/config/static_config.yaml")
    search_config = load_config(search_config_path)
    static_config = load_config(static_config_path)

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = DataLoaderManager.setup_dataloaders(
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
        config=search_config,
        study_name="cnn_hyperparameter_search",
        direction=static_config["direction"],
    )

    search.run()


if __name__ == "__main__":
    main()
