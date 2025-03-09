from tqdm import tqdm
import torch
from dc1.net import Net
from dc1.batch_sampler import BatchSampler
from typing import Callable, List
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
from tabulate import tabulate


def train_model(
        model: Net,
        train_sampler: BatchSampler,
        optimizer: torch.optim.Optimizer,
        loss_function: Callable[..., torch.Tensor],
        device: str,
) -> List[torch.Tensor]:
    # Lets keep track of all the losses:
    losses = []
    # Put the model in train mode:
    model.train()
    # Feed all the batches one by one:
    for batch in tqdm(train_sampler):
        # Get a batch:
        x, y = batch
        # Making sure our samples are stored on the same device as our model:
        x, y = x.to(device), y.to(device)
        # Get predictions:
        predictions = model.forward(x)
        loss = loss_function(predictions, y)
        losses.append(loss)
        # We first need to make sure we reset our optimizer at the start.
        # We want to learn from each batch seperately,
        # not from the entire dataset at once.
        optimizer.zero_grad()
        # We now backpropagate our loss through our model:
        loss.backward()
        # We then make the optimizer take a step in the right direction.
        optimizer.step()
    return losses


def test_model(
        model: Net,
        test_sampler: BatchSampler,
        loss_function: Callable[..., torch.Tensor],
        device: str,
) -> List[torch.Tensor]:
    # Setting the model to evaluation mode:
    model.eval()
    losses = []
    all_preds = []
    all_labels = []
    # We need to make sure we do not update our model based on the test data:
    with torch.no_grad():
        for (x, y) in tqdm(test_sampler):
            # Making sure our samples are stored on the same device as our model:
            x = x.to(device)
            y = y.to(device)
            prediction = model.forward(x)
            loss = loss_function(prediction, y)
            losses.append(loss)

            predicted_classes = prediction.argmax(dim=1)  # Get class with highest probability
            all_preds.extend(predicted_classes.cpu().numpy())  # Collect all predictions
            all_labels.extend(y.cpu().numpy())  # Collect all ground-truth labels

    conf_matrix = confusion_matrix(all_labels, all_preds)

    return losses, conf_matrix

def compute_metrics(conf_matrix):
    num_classes = conf_matrix.shape[0]  # Get number of classes
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)
    accuracy = np.zeros(num_classes)

    for i in range(num_classes):
        TP = conf_matrix[i, i]  # True Positives
        FP = conf_matrix[:, i].sum() - TP  # False Positives
        FN = conf_matrix[i, :].sum() - TP  # False Negatives
        TN = conf_matrix.sum() - (TP + FP + FN)  # True Negatives

        precision[i] = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall[i] = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
        accuracy[i] = (TP + TN) / conf_matrix.sum()

    return precision, recall, f1_score, accuracy

def print_confusion_matrix(conf_matrix, class_names=None):
    class_names = ["Atelectasis", "Effusion", "Infiltration", "No finding", "Nodule", "Pneumothorax"]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(conf_matrix))]
    table = [[class_names[i]] + list(row) for i, row in enumerate(conf_matrix)]
    print("\nConfusion Matrix:\n")
    print(tabulate(table, headers=["Class ↓ / Predicted →"] + class_names, tablefmt="grid"))