import torch
from pathlib import Path
import argparse
import json
import os

from src.models.cnn import CNN
from src.dataset.data_loader_manager import DataLoaderManager
from src.ethics.cam import generate_cam
from src.ethics.saliency_map import generate_saliency_map
from src.ethics.gradcam import generate_gradcam
from src.utils.config_loader import load_config

def load_model(model_path: Path, params: dict, num_classes: int, input_size: int, device: torch.device):
    model = CNN(params=params, num_classes=num_classes, input_size=input_size)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Visualize CAM and saliency maps from saved model.")
    parser.add_argument("--trial", type=int, required=True, help="Trial number of the saved model to visualize")
    args = parser.parse_args()
    os.makedirs("cam_outputs", exist_ok=True)

    trial_number = args.trial
    print(f"Selected trial: {trial_number}")

    static_config = load_config(Path("src/config/static_config.yaml"))
    
    model_path = Path(f"saved_outputs/completed/trial_{trial_number}/model.pt")
    params_path = Path(f"saved_outputs/completed/trial_{trial_number}/info.json")

    with open(params_path, "r") as f:
        data = json.load(f)

    params = data.get("params")

    device = (
        torch.device("mps") if torch.backends.mps.is_available() and torch.backends.mps.is_built()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    model = load_model(model_path, params, static_config["num_classes"], static_config["input_size"], device)

    _, _, test_loader = DataLoaderManager.setup_dataloaders(
        x_train_path=Path(static_config["train_data"]),
        y_train_path=Path(static_config["train_labels"]),
        x_test_path=Path(static_config["test_data"]),
        y_test_path=Path(static_config["test_labels"]),
        batch_size=static_config["batch_size"],
    )

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

            _, logits = model(img_input)
            predicted_class = torch.argmax(logits, dim=1).item()

            print(f"Visualizing CAM for true class: {label}, predicted: {predicted_class}")
            generate_cam(model, img_input, class_index=predicted_class, true_class=label)
            generate_saliency_map(model, img_input.clone().detach().requires_grad_(True), class_index=predicted_class)
            generate_gradcam(model, img_input, class_index=predicted_class, true_class=label)

            if predicted_class == label:
                cor += 1
            tot += 1
            seen_classes.add(label)

            if len(seen_classes) == max_classes:
                break
        if len(seen_classes) == max_classes:
            break

    accuracy = 100 * cor / tot
    print(f"\nCorrect predictions: {cor}/{tot} ({accuracy:.2f}%)")

if __name__ == "__main__":
    main()
