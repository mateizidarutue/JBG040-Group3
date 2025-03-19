import numpy as np
import torch
from typing import Tuple
from pathlib import Path
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, x: Path, y: Path) -> None:
        self.imgs = np.load(x)
        self.targets = np.load(y)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        image = torch.from_numpy(self.imgs[idx] / 255).float() * 2 - 1
        label = self.targets[idx]
        return image, label
