import torch
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from src.dataset.image_dataset import ImageDataset
import os


class DataLoaderManager:
    @staticmethod
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
        sample_weights = class_weights[
            full_train_dataset.targets[train_dataset.indices]
        ]

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
