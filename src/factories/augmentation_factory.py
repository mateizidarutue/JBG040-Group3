import torchvision.transforms as transforms
from typing import Dict, Any
from torchvision.transforms import functional as F
from PIL import Image


class AugmentationFactory:
    @staticmethod
    def get_augmentations(
        config: Dict[str, Any], input_size: int
    ) -> transforms.Compose:
        augmentations = []

        if config["rotation_enabled"]:
            augmentations.append(transforms.RandomRotation(degrees=(-5, 5)))

        augmentations.append(FixedLungCrop())
        augmentations.append(transforms.Resize(input_size))

        if config["brightness_enabled"] or config["contrast_enabled"]:
            brightness = config["brightness"] if config["brightness_enabled"] else 0
            contrast = config["contrast"] if config["contrast_enabled"] else 0
            augmentations.append(
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=brightness, contrast=contrast)],
                    p=0.4,
                )
            )

        augmentations.append(transforms.Resize(input_size))

        return transforms.Compose(augmentations)


class FixedLungCrop:
    def __call__(self, img: Image.Image) -> Image.Image:
        return F.crop(img, top=10, left=12, height=98, width=104)
