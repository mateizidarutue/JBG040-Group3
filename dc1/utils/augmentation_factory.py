import torchvision.transforms as transforms

class AugmentationFactory:
    @staticmethod
    def get_augmentations(config):
        augmentations = []

        if config["use_flip"]:
            flip_prob = config["flip_prob"]
            augmentations.append(transforms.RandomHorizontalFlip(p=flip_prob))

        if config["use_crop"]:
            crop_size = config["crop_size"]
            augmentations.append(transforms.RandomResizedCrop(size=crop_size))

        if config["use_brightness"] or config["use_contrast"]:
            brightness = config["brightness"] if config["use_brightness"] else 0
            contrast = config["contrast"] if config["use_contrast"] else 0
            augmentations.append(transforms.ColorJitter(brightness=brightness, contrast=contrast))

        if config["use_upscale"]:
            upscale = config["upscale"]
            upscale_mode = config["upscale_mode"]
            augmentations.append(
                transforms.Resize(
                    size=(int(upscale * 100), int(upscale * 100)), 
                    interpolation=AugmentationFactory.get_interpolation(upscale_mode)
                )
            )

        return transforms.Compose(augmentations)

    @staticmethod
    def get_interpolation(mode):
        if mode == 'bilinear':
            return transforms.InterpolationMode.BILINEAR
        elif mode == 'nearest':
            return transforms.InterpolationMode.NEAREST
        elif mode == 'cubic':
            return transforms.InterpolationMode.BICUBIC
        else:
            raise ValueError(f"Unsupported interpolation mode: {mode}")
