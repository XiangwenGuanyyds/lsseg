"""Earlier experiment: spatial attention and learned feature gating on FPN maps.

Unused in the final model.
"""

import torch
import torch.nn as nn
from utils import MODELS


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1

        self.attn_conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attn_input = torch.cat([avg_map, max_map], dim=1)   # [B, 2, H, W]
        attn_map = self.sigmoid(self.attn_conv(attn_input))     # [B, 1, H, W]
        return x * attn_map, attn_map


class GateMap(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.gate_conv(x)   # [B, 1, H, W]


@MODELS.register_module()
class OcclusionAwareAttention(nn.Module):
    def __init__(self, in_channels=256, spatial_kernel=7):
        super().__init__()

        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel)

        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        )

        self.gate_map = GateMap(in_channels)

    def forward(self, features):
        new_features = {}

        for k, v in features.items():
            attn_feat, _ = self.spatial_attention(v)     # 空间注意力后的特征
            refined_feat = self.refine_conv(attn_feat)   # 局部增强特征
            gate = self.gate_map(refined_feat)           # [B,1,H,W]

            out = v + gate * refined_feat
            new_features[k] = out

        return new_features
