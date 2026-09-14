"""Convolutional Block Attention Module (CBAM).

Earlier FPN attention experiment.

Channel and spatial attention from:

    Sanghyun Woo, Jongchan Park, Joon-Young Lee, and In So Kweon.
    "CBAM: Convolutional Block Attention Module." ECCV 2018.

Applied to each FPN level, between the backbone and RPN/RoI heads.
"""

import torch
import torch.nn as nn
from utils import MODELS


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        hidden = max(in_channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        attn = self.sigmoid(avg_out + max_out)
        return x * attn


class CBAMSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attn_input = torch.cat([avg_map, max_map], dim=1)
        attn = self.sigmoid(self.conv(attn_input))
        return x * attn


@MODELS.register_module()
class CBAM(nn.Module):
    """CBAM (Woo et al., ECCV 2018).

    F'  = Mc(F) ⊗ F
    F'' = Ms(F') ⊗ F'

    Channel attention, then spatial attention. Element-wise weighting.
    """

    def __init__(self, in_channels=256, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction=reduction)
        self.spatial_attention = CBAMSpatialAttention(kernel_size=spatial_kernel)

    def forward(self, features):
        new_features = {}
        for k, v in features.items():
            x = self.channel_attention(v)
            x = self.spatial_attention(x)
            new_features[k] = x
        return new_features
