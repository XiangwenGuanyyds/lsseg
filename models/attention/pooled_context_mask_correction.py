"""Pooled neighboring-RoI context for mask-logit correction.

Earlier experiment; unused in the final model.

The standard mask branch remains the main prediction path. This optional
branch pools each RoI to one vector, combines neighboring RoI appearance
features with relative box geometry, and predicts a spatial mask-logit
correction for the target RoI.
"""

import torch
import torch.nn as nn
from torchvision.ops import box_iou

from utils import MODELS


@MODELS.register_module()
class PooledContextMaskCorrection(nn.Module):
    """Predict a mask correction from pooled neighboring RoI context."""

    def __init__(
        self,
        in_channels=256,
        context_dim=64,
        relation_hidden_dim=64,
        neighbor_top_k=4,
        neighbor_iou_threshold=0.1,
        delta_scale=1.0,
        foreground_class=1,
    ):
        super().__init__()
        if in_channels <= 0 or context_dim <= 0 or relation_hidden_dim <= 0:
            raise ValueError("Feature dimensions must be positive.")
        if neighbor_top_k <= 0:
            raise ValueError("neighbor_top_k must be positive.")
        if not 0 <= neighbor_iou_threshold <= 1:
            raise ValueError("neighbor_iou_threshold must be in [0, 1].")

        self.in_channels = int(in_channels)
        self.context_dim = int(context_dim)
        self.neighbor_top_k = int(neighbor_top_k)
        self.neighbor_iou_threshold = float(neighbor_iou_threshold)
        self.delta_scale = float(delta_scale)
        self.foreground_class = int(foreground_class)

        self.pool = nn.AdaptiveMaxPool2d(1)
        self.relation = nn.Sequential(
            nn.Linear(2 * self.in_channels + 4, int(relation_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(relation_hidden_dim), self.context_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(
                self.in_channels + self.context_dim,
                self.context_dim,
                kernel_size=1,
                padding=0,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.context_dim, self.context_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.context_dim, 1, kernel_size=1),
        )
        # Keep the optional branch initially identical to the standard mask
        # path. The branch learns a correction during subsequent training.
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def _select_neighbors(self, boxes):
        count = boxes.shape[0]
        if count <= 1:
            return [[] for _ in range(count)]

        overlaps = box_iou(boxes, boxes)
        index = torch.arange(count, device=boxes.device)
        neighbors = []
        for i in range(count):
            candidates = torch.where(
                (overlaps[i] >= self.neighbor_iou_threshold) & (index != i)
            )[0]
            if candidates.numel() == 0:
                neighbors.append([])
                continue
            order = torch.argsort(overlaps[i, candidates], descending=True)
            neighbors.append(candidates[order[: self.neighbor_top_k]].tolist())
        return neighbors

    @staticmethod
    def _relative_geometry(target_box, neighbor_box):
        target_xy = (target_box[:2] + target_box[2:]) * 0.5
        neighbor_xy = (neighbor_box[:2] + neighbor_box[2:]) * 0.5
        target_size = (target_box[2:] - target_box[:2]).clamp_min(1e-6)
        neighbor_size = (neighbor_box[2:] - neighbor_box[:2]).clamp_min(1e-6)

        center_delta = (neighbor_xy - target_xy) / target_size
        size_ratio = torch.log(neighbor_size / target_size)
        return torch.cat((center_delta, size_ratio), dim=0)

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
            pooled = self.pool(image_features).flatten(1)  # [R, C]
            neighbor_lists = self._select_neighbors(image_boxes)
            image_outputs = []

            for i, neighbor_ids in enumerate(neighbor_lists):
                if neighbor_ids:
                    target = pooled[i].expand(len(neighbor_ids), -1)
                    neighbor = pooled[neighbor_ids]
                    geometry = torch.stack(
                        [
                            self._relative_geometry(image_boxes[i], image_boxes[j])
                            for j in neighbor_ids
                        ],
                        dim=0,
                    )
                    relation_input = torch.cat((target, neighbor, geometry), dim=1)
                    context = self.relation(relation_input).mean(dim=0)
                else:
                    context = pooled.new_zeros((self.context_dim,))

                context_map = context.view(1, self.context_dim, 1, 1).expand(
                    1, self.context_dim, image_features.shape[-2], image_features.shape[-1]
                )
                target_feature = image_features[i : i + 1]
                correction_input = torch.cat((target_feature, context_map), dim=1)
                image_outputs.append(self.decoder(correction_input))

            outputs.append(torch.cat(image_outputs, dim=0))
            start += count

        return self.delta_scale * torch.cat(outputs, dim=0)
