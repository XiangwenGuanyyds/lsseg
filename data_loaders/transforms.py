"""Apply configured transforms to images and their target fields."""

import random

import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

from utils import TRANSFORMS


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
            # print(f"[Compose] transform={type(t).__name__}, image={getattr(image, 'shape', image.size)}")
        return image, target


# Geometric transforms update all spatial target fields.
@TRANSFORMS.register_module()
class Resize:
    """Resize to ``size`` (H, W) and rescale spatial target fields."""

    def __init__(self, size):
        self.size = size  # torchvision uses (height, width)

    def __call__(self, image, target):
        w_old, h_old = image.size
        sw, sh = self.size[1] / w_old, self.size[0] / h_old
        # print(f"[Resize] input_size={(h_old, w_old)}, output_size={tuple(self.size)}, scale={(sh, sw)}")

        image = F.resize(image, self.size)

        if "boxes" in target:
            target["boxes"][:, [0, 2]] *= sw
            target["boxes"][:, [1, 3]] *= sh

        # Nearest-neighbour interpolation preserves binary mask values.
        if "masks" in target and target["masks"].numel() > 0:
            masks = target["masks"].unsqueeze(1).float()
            masks = F.resize(masks, self.size, interpolation=InterpolationMode.NEAREST)
            target["masks"] = masks.squeeze(1).byte()

        if "inner_centers" in target and target["inner_centers"].numel() > 0:
            target["inner_centers"][:, 0] *= sw
            target["inner_centers"][:, 1] *= sh

        return image, target


@TRANSFORMS.register_module()
class RandomHorizontalFlip:
    """Mirror image + boxes + masks + inner_centers with probability ``p``."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target

        w, _ = image.size
        image = F.hflip(image)
        # print(f"[RandomHorizontalFlip] image_size={image.size}, instances={len(target['labels'])}")

        if "boxes" in target and target["boxes"].numel() > 0:
            boxes = target["boxes"]
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target["boxes"] = boxes

        if "masks" in target and target["masks"].numel() > 0:
            target["masks"] = target["masks"].flip(-1)

        if "inner_centers" in target and target["inner_centers"].numel() > 0:
            target["inner_centers"][:, 0] = w - 1 - target["inner_centers"][:, 0]

        return image, target


# Photometric transforms update the image only.
@TRANSFORMS.register_module()
class RandomBrightnessContrast:
    def __init__(self, brightness=0.2, contrast=0.2, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = F.adjust_brightness(image, 1 + random.uniform(-self.brightness, self.brightness))
            image = F.adjust_contrast(image, 1 + random.uniform(-self.contrast, self.contrast))
        return image, target


@TRANSFORMS.register_module()
class GaussianBlur:
    def __init__(self, kernel_size=5, sigma=(0.1, 2.0), p=0.2):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            image = F.gaussian_blur(image, self.kernel_size, [sigma, sigma])
        return image, target


# Tensor conversion runs at the end of the configured pipeline.
@TRANSFORMS.register_module()
class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        # print(f"[ToTensor] image_shape={tuple(image.shape)}, dtype={image.dtype}, range=({image.min().item():.4f}, {image.max().item():.4f})")
        return image, target
