"""Earlier comparison using torchvision's complete Mask R-CNN forward path.

The thesis baseline and final model use models.mask_rcnn.MaskRCNN.
"""

import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn

from utils import MODELS


@MODELS.register_module()
class NativeMaskRCNN(nn.Module):
    """Drop-in torchvision Mask R-CNN."""

    def __init__(self, cfg):
        super().__init__()
        num_classes = cfg.MODEL["num_classes"]
        full_maskrcnn_weights = cfg.MODEL["full_maskrcnn_weights"]
        resnet50_body_weights = cfg.MODEL["resnet50_body_weights"]
        trainable_resnet_layers = cfg.MODEL["trainable_resnet_layers"]
        self.model = maskrcnn_resnet50_fpn(
            weights=full_maskrcnn_weights,
            weights_backbone=resnet50_body_weights,
            trainable_backbone_layers=trainable_resnet_layers,
            num_classes=num_classes,
        )

    def forward(self, x, targets=None):
        return self.model(x, targets)
