"""Pixel-level context correction for the Mask R-CNN mask branch.

Earlier experiment; unused in the final model.

The module keeps the standard mask head as the main prediction path. For each
RoI, its spatial RoI features provide the queries, while spatial features from
nearby RoIs provide keys and values. A relative-position bias is added to the
attention scores, and a small decoder predicts a per-pixel mask-logit
correction.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou

from utils import MODELS


@MODELS.register_module()
class PixelContextMaskCorrection(nn.Module):
    """Predict a spatial mask-logit correction from nearby RoI context."""

    def __init__(
        self,
        in_channels=256,
        token_dim=128,
        num_heads=4,
        neighbor_top_k=4,
        neighbor_iou_threshold=0.1,
        relative_hidden_dim=64,
        delta_scale=1.0,
        foreground_class=1,
    ):
        super().__init__()
        if token_dim <= 0 or token_dim % num_heads != 0:
            raise ValueError("token_dim must be positive and divisible by num_heads.")
        if neighbor_top_k <= 0:
            raise ValueError("neighbor_top_k must be positive.")
        if neighbor_iou_threshold < 0 or neighbor_iou_threshold > 1:
            raise ValueError("neighbor_iou_threshold must be in [0, 1].")

        self.in_channels = int(in_channels)
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.token_dim // self.num_heads
        self.neighbor_top_k = int(neighbor_top_k)
        self.neighbor_iou_threshold = float(neighbor_iou_threshold)
        self.delta_scale = float(delta_scale)
        self.foreground_class = int(foreground_class)

        self.norm = nn.LayerNorm(self.in_channels)
        self.query = nn.Linear(self.in_channels, self.token_dim)
        self.key = nn.Linear(self.in_channels, self.token_dim)
        self.value = nn.Linear(self.in_channels, self.token_dim)
        self.output = nn.Linear(self.token_dim, self.token_dim)

        self.relative_position = nn.Sequential(
            nn.Linear(2, int(relative_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(relative_hidden_dim), self.num_heads),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(self.token_dim, self.token_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.token_dim, 1, kernel_size=1),
        )
        # Start with no correction so the standard mask branch remains the
        # initial reference path when this optional branch is enabled.
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    @staticmethod
    def _grid_centers(box, height, width):
        x1, y1, x2, y2 = box.unbind()
        xs = x1 + (torch.arange(width, device=box.device, dtype=box.dtype) + 0.5) / width * (x2 - x1)
        ys = y1 + (torch.arange(height, device=box.device, dtype=box.dtype) + 0.5) / height * (y2 - y1)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)

    def _select_neighbors(self, boxes):
        count = boxes.shape[0]
        if count <= 1:
            return [[] for _ in range(count)]

        overlaps = box_iou(boxes, boxes)
        neighbors = []
        for i in range(count):
            candidates = torch.where(
                (overlaps[i] >= self.neighbor_iou_threshold)
                & (torch.arange(count, device=boxes.device) != i)
            )[0]
            if candidates.numel() == 0:
                neighbors.append([])
                continue
            order = torch.argsort(overlaps[i, candidates], descending=True)
            neighbors.append(candidates[order[: self.neighbor_top_k]].tolist())
        return neighbors

    def _attend_one(
        self, q, k, v, target_box, target_pos, neighbor_positions, height, width
    ):
        context_pos = torch.cat(neighbor_positions, dim=0)
        target_size = (target_box[2:] - target_box[:2]).clamp_min(1e-6)

        q = q.view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        relative = (context_pos.unsqueeze(0) - target_pos.unsqueeze(1)) / target_size
        position_bias = self.relative_position(relative)  # [L, M*L, heads]
        position_bias = position_bias.permute(2, 0, 1)  # [heads, L, M*L]

        weights = torch.softmax(scores + position_bias, dim=-1)
        context = torch.matmul(weights, v)
        context = context.transpose(0, 1).contiguous().view(-1, self.token_dim)
        context = self.output(context)
        context = context.transpose(0, 1).reshape(1, self.token_dim, height, width)
        return self.decoder(context)

    def forward(self, x, roi_counts, roi_boxes):
        if x.dim() != 4:
            raise ValueError(f"Expected [N, C, H, W] RoI features, got {tuple(x.shape)}.")
        if sum(int(v) for v in roi_counts) != x.shape[0]:
            raise ValueError("roi_counts must sum to the number of RoI features.")
        if len(roi_counts) != len(roi_boxes):
            raise ValueError("roi_counts and roi_boxes must have the same length.")

        if x.shape[0] == 0:
            return x.new_zeros((0, 1, x.shape[-2], x.shape[-1]))

        outputs = []
        start = 0
        for count, boxes in zip(roi_counts, roi_boxes):
            count = int(count)
            if count == 0:
                continue
            image_features = x[start : start + count]
            image_boxes = boxes.to(device=x.device, dtype=x.dtype)
            neighbor_lists = self._select_neighbors(image_boxes)

            # Project every RoI once. The previous implementation recomputed
            # the same neighbor K/V projections for every target RoI.
            _, height, width = image_features.shape[1:]
            tokens = image_features.flatten(2).transpose(1, 2)  # [R, L, C]
            tokens = self.norm(tokens)
            q_all = self.query(tokens)  # [R, L, D]
            k_all = self.key(tokens)    # [R, L, D]
            v_all = self.value(tokens)  # [R, L, D]
            positions = [
                self._grid_centers(box, height, width)
                for box in image_boxes
            ]

            image_outputs = []
            for i, neighbor_ids in enumerate(neighbor_lists):
                if not neighbor_ids:
                    image_outputs.append(
                        x.new_zeros((1, 1, x.shape[-2], x.shape[-1]))
                    )
                    continue
                neighbor_keys = [k_all[j] for j in neighbor_ids]
                neighbor_values = [v_all[j] for j in neighbor_ids]
                neighbor_positions = [positions[j] for j in neighbor_ids]
                image_outputs.append(
                    self._attend_one(
                        q_all[i],
                        torch.cat(neighbor_keys, dim=0),
                        torch.cat(neighbor_values, dim=0),
                        image_boxes[i],
                        positions[i],
                        neighbor_positions,
                        height,
                        width,
                    )
                )
            outputs.append(torch.cat(image_outputs, dim=0))
            start += count

        return self.delta_scale * torch.cat(outputs, dim=0)
