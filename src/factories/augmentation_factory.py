import torchvision.transforms as transforms
from typing import Dict, Any


class AugmentationFactory:
    @staticmethod
    def get_augmentations(
        config: Dict[str, Any], input_size: int
    ) -> transforms.Compose:
        augmentations = []

        if config["rotation_enabled"]:
            augmentations.append(transforms.RandomRotation(degrees=(-5, 5)))

        if config["crop_enabled"]:
            crop_size = config["crop_size"]
            augmentations.append(
                transforms.RandomApply(
                    [transforms.RandomResizedCrop(size=crop_size)], p=0.3
                )
            )

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
