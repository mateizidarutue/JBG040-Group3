import torchvision.transforms as transforms

class AugmentationFactory:
    @staticmethod
    def get_augmentations(config):
        augmentations = []

        if config.get("use_flip"):
            flip_prob = config.get("flip_prob")
            augmentations.append(transforms.RandomHorizontalFlip(p=flip_prob))

        if config.get("use_crop"):
            crop_size = config.get("crop_size")
            augmentations.append(transforms.RandomResizedCrop(size=crop_size))

        if config.get("use_brightness") or config.get("use_contrast"):
            brightness = config.get("brightness") if config.get("use_brightness") else 0
            contrast = config.get("contrast") if config.get("use_contrast") else 0
            augmentations.append(transforms.ColorJitter(brightness=brightness, contrast=contrast))


        if config.get("use_upscale"):
            upscale = config.get("upscale")
            upscale_mode = config.get("upscale_mode", "bilinear")
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
