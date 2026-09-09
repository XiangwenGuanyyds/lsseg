"""Experimental box refinement, not used in the final method.

This module tests whether overlap between proposals can help box localisation
when instances are close together. It collects overlap statistics for each
proposal, maps them through an MLP, and adds the result to the box regression
feature with a learnable scale. The classification path stays unchanged.

"""

import torch
import torch.nn as nn
from torchvision.ops import box_iou

from utils import MODELS


@MODELS.register_module(name="OverlapAwareBoxRefinement")
class OverlapAwareBoxRefinement(nn.Module):
    def __init__(
        self,
        feature_dim=1024,
        context_hidden_dim=128,
        thresholds=(0.5, 0.7),
        residual_init=0.0,
    ):
        super().__init__()
        self.thresholds = tuple(float(t) for t in thresholds)
        context_dim = 2 + len(self.thresholds)
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, context_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(context_hidden_dim, feature_dim),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))

    def forward(self, box_feature, proposals):
        if proposals is None:
            raise ValueError("OverlapAwareBoxRefinement requires per-image proposals.")

        context = self._proposal_overlap_context(proposals, box_feature.device, box_feature.dtype)
        return box_feature + self.residual_scale * self.mlp(context)

    def _proposal_overlap_context(self, proposals, device, dtype):
        rows = []
        for boxes in proposals:
            boxes = boxes.to(device=device, dtype=dtype)
            n = int(boxes.shape[0])
            if n == 0:
                continue
            if n == 1:
                rows.append(torch.zeros((1, 2 + len(self.thresholds)), device=device, dtype=dtype))
                continue

            ious = box_iou(boxes, boxes)
            ious.fill_diagonal_(0.0)

            max_iou = ious.max(dim=1).values
            mean_iou = ious.sum(dim=1) / float(n - 1)
            denom = torch.log1p(torch.tensor(float(n - 1), device=device, dtype=dtype))

            parts = [max_iou[:, None], mean_iou[:, None]]
            for threshold in self.thresholds:
                count = (ious > threshold).sum(dim=1).to(dtype=dtype)
                parts.append((torch.log1p(count) / denom)[:, None])
            rows.append(torch.cat(parts, dim=1))

        if not rows:
            return torch.empty((0, 2 + len(self.thresholds)), device=device, dtype=dtype)
        return torch.cat(rows, dim=0)
