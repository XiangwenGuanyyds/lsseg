"""Stock torchvision Mask R-CNN — sanity baseline for the decomposed Architecture.

Backbone is initialized from ImageNet-pretrained ResNet50; everything else
(RPN, RoI heads, mask FCN) is random init. This matches the standard
training-from-scratch setup in Detectron2 / mmdetection / the Mask R-CNN
paper, and aligns with the decomposed Architecture which also uses
ImageNet-only pretraining.

If the AP this produces differs from the decomposed Architecture pipeline
by more than expected seed/training noise, the decomposition has a bug.
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
