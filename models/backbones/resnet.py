import torch.nn as nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from utils import MODELS


@MODELS.register_module()
class ResNet50FPN(nn.Module):
    """ResNet50 + FPN backbone (torchvision wrapper)."""

    def __init__(self, weights):
        super().__init__()
        self.body = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=weights,
        )
        self.out_channels = self.body.out_channels
        # print("[debug] backbone weights:", weights, "out channels:", self.out_channels)

    def forward(self, x):
        # print("[debug] input shape:", tuple(x.shape), "device:", x.device)
        features = self.body(x)
        # print("[debug] FPN output shapes:", {name: tuple(value.shape) for name, value in features.items()})
        # FPN output shapes:
        # {'0': (1, 256, 200, 200), '1': (1, 256, 100, 100), '2': (1, 256, 50, 50), '3': (1, 256, 25, 25), 'pool': (1, 256, 13, 13)}
        return features
