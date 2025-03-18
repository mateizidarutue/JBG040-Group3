import numpy as np
import torch
from typing import Tuple, Dict, Any
from pathlib import Path
from torch.utils.data import Dataset
from dc1.utils.augmentation_factory import AugmentationFactory


class ImageDataset(Dataset):
    def __init__(self, x: Path, y: Path, config: Dict[str, Any]) -> None:
        self.imgs = np.load(x)
        self.targets = np.load(y)
        self.transform = AugmentationFactory.get_augmentations(config)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        image = torch.from_numpy(self.imgs[idx] / 255).float()
        image = self.transform(image)
        label = self.targets[idx]
        return image, label
