import numpy as np
import torch
from typing import Tuple
from pathlib import Path
from torch.utils.data import Dataset
from torch import Tensor


class ImageDataset(Dataset):
    def __init__(self, x: Path, y: Path) -> None:
        self.imgs = np.load(x)
        self.targets = np.load(y)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        image = torch.from_numpy(self.imgs[idx] / 255).float() * 2 - 1
        label = torch.tensor(self.targets[idx], dtype=torch.long)
        return image, label
