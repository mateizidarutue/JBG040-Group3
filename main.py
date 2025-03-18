import torch
import yaml
from pathlib import Path
from dc1.trainer.trainer import Trainer
from dc1.search.optuna_bayes_hyperband import OptunaBayesHyperband
from dc1.dataset.image_dataset import ImageDataset


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config_path = "config.yaml"
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data_path = Path(config["dataset"]["train_data"])
    train_labels_path = Path(config["dataset"]["train_labels"])
    dataset = ImageDataset(train_data_path, train_labels_path)

    trainer = Trainer(
        config=config,
        dataset=dataset,
        batch_size=config["batch_size"],
        device=device,
    )

    search = OptunaBayesHyperband(
        min_budget=config["min_budget"],
        max_budget=config["max_epochs"],
        eta=config["eta"],
        trials_per_batch=config["trials_per_batch"],
        number_of_runs=config["number_of_runs"],
        training_fn=trainer.train,
        config_path=config_path,
        study_name="cnn_hyperparameter_search",
        direction=config["direction"],
    )

    search.run()

    search.save_results("results.json")


if __name__ == "__main__":
    main()
