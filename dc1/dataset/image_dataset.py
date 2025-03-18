import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from typing import Tuple
from pathlib import Path


class ImageDataset(Dataset):
    def __init__(self, x: Path, y: Path) -> None:
        self.targets = self.load_numpy_arr_from_npy(y)
        self.imgs = self.load_numpy_arr_from_npy(x)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        image = torch.from_numpy(self.imgs[idx] / 255).float()
        label = self.targets[idx]
        return image, label

    @staticmethod
    def load_numpy_arr_from_npy(path: Path) -> np.ndarray:
        return np.load(path)

    @staticmethod
    def split(dataset, split_ratio: float):
        train_size = int((1 - split_ratio) * len(dataset))
        val_size = len(dataset) - train_size
        return random_split(dataset, [train_size, val_size])
