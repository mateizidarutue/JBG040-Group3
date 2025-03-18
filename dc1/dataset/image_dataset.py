import numpy as np
import torch
from typing import Tuple
from pathlib import Path


class ImageDataset:
    def __init__(self, x: Path, y: Path) -> None:
        self.targets = ImageDataset.load_numpy_arr_from_npy(y)
        self.imgs = ImageDataset.load_numpy_arr_from_npy(x)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        image = torch.from_numpy(self.imgs[idx] / 255).float()
        label = self.targets[idx]
        return image, label

    @staticmethod
    def load_numpy_arr_from_npy(path: Path) -> np.ndarray:
        return np.load(path)
