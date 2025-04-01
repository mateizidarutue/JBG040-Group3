from pathlib import Path
import torch
import yaml

from src.trainer.trainer import Trainer
from src.search.optuna_bayes_hyperband import OptunaBayesHyperband
from src.dataset.data_loader_manager import DataLoaderManager
from src.ethics.cam import generate_cam
from src.ethics.saliency_map import generate_saliency_map

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

    #sample_image, sample_label = next(iter(test_loader))
    #sample_image = sample_image[0].unsqueeze(0).to(device)

    best_model = trainer._model
    best_model.to(device)
    best_model.eval()

    seen_classes = set()
    cor = 0
    tot = 0
    max_classes = static_config["num_classes"]

    for images, labels in test_loader:
        for img, label in zip(images, labels):
            label = label.item()
            if label in seen_classes:
                continue

            img_input = img.unsqueeze(0).to(device)

            features, logits = best_model(img_input, return_features=True)
            predicted_class = torch.argmax(logits, dim=1).item()

            print(f"Visualizing CAM for true class: {label}, predicted: {predicted_class}")
            generate_cam(best_model, img_input, class_index=predicted_class, true_class=label)

            # Generate Saliency Map
            saliency_input = img_input.clone().detach().requires_grad_(True)
            generate_saliency_map(best_model, saliency_input, class_index=predicted_class)

            if predicted_class == label:
                cor += 1
            tot += 1
            seen_classes.add(label)

            if len(seen_classes) == max_classes:
                break
        if len(seen_classes) == max_classes:
            break

        accuracy = 100 * cor / tot
        print(f"\n Correct predictions: {cor}/{tot} ({accuracy:.2f}%)")

if __name__ == "__main__":
    main()
