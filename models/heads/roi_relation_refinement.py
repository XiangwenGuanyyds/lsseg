"""RoI relation refinement for box-branch spatial features.

The module is optional and is only constructed when the config provides
``cfg.MODEL["box_roi_relation_refinement"]``. It builds per-image RoI
relations from proposal geometry and pooled RoI features, then injects the
aggregated neighbour context before the box head flattens the RoI features.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou

from utils import MODELS


@MODELS.register_module(name="RoIRelationBoxFeatureRefinement")
class RoIRelationBoxFeatureRefinement(nn.Module):
    def __init__(
        self,
        in_channels=256,
        relation_dim=128,
        geom_hidden_dim=64,
        top_k=32,
        residual_init=0.1,
        edge_loss_weight=0.0,
        max_edge_pairs=4096,
    ):
        super().__init__()
        self.top_k = int(top_k) if top_k is not None else None
        self.edge_loss_weight = float(edge_loss_weight)
        self.max_edge_pairs = int(max_edge_pairs) if max_edge_pairs is not None else None
        self.last_edge_loss = None

        self.query = nn.Linear(in_channels, relation_dim)
        self.key = nn.Linear(in_channels, relation_dim)
        self.value = nn.Linear(in_channels, relation_dim)
        self.output = nn.Linear(relation_dim, in_channels)
        self.geometry_bias = nn.Sequential(
            nn.Linear(5, geom_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(geom_hidden_dim, 1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))

    def forward(self, box_features, proposals, matched_idxs=None, labels=None):
        if proposals is None:
            raise ValueError("RoIRelationBoxFeatureRefinement requires per-image proposals.")
        self.last_edge_loss = None
        if box_features.numel() == 0:
            return box_features

        pooled = F.adaptive_avg_pool2d(box_features, output_size=1).flatten(1)
        contexts = []
        edge_losses = []
        start = 0
        for img_id, boxes in enumerate(proposals):
            n = int(boxes.shape[0])
            if n == 0:
                continue
            f = pooled[start:start + n]
            img_labels = labels[img_id] if labels is not None else None
            img_matched = matched_idxs[img_id] if matched_idxs is not None else None
            context, edge_loss = self._image_context(
                f,
                boxes.to(device=f.device, dtype=f.dtype),
                labels=img_labels,
                matched_idxs=img_matched,
            )
            contexts.append(context)
            if edge_loss is not None:
                edge_losses.append(edge_loss)
            start += n

        if not contexts:
            return box_features
        if edge_losses:
            self.last_edge_loss = torch.stack(edge_losses).mean() * self.edge_loss_weight

        context = torch.cat(contexts, dim=0)
        context = self.output(context).view(box_features.shape[0], box_features.shape[1], 1, 1)
        return box_features + self.residual_scale * context

    def _image_context(self, roi_features, boxes, labels=None, matched_idxs=None):
        n = int(roi_features.shape[0])
        if n == 1:
            return roi_features.new_zeros((1, self.value.out_features)), None

        q = self.query(roi_features)
        k = self.key(roi_features)
        v = self.value(roi_features)

        scores = torch.matmul(q, k.transpose(0, 1)) / math.sqrt(float(q.shape[1]))
        scores = scores + self.geometry_bias(self._pairwise_geometry(boxes)).squeeze(-1)

        eye = torch.eye(n, device=scores.device, dtype=torch.bool)
        edge_loss = self._edge_loss(scores, labels, matched_idxs, eye)
        scores = scores.masked_fill(eye, torch.finfo(scores.dtype).min)

        if self.top_k is not None and 0 < self.top_k < n - 1:
            vals, inds = torch.topk(scores, k=self.top_k, dim=1)
            weights = F.softmax(vals, dim=1)
            selected_v = v[inds]
            return (weights.unsqueeze(-1) * selected_v).sum(dim=1), edge_loss

        weights = F.softmax(scores, dim=1)
        return torch.matmul(weights, v), edge_loss

    def _edge_loss(self, scores, labels, matched_idxs, eye):
        if self.edge_loss_weight <= 0.0 or labels is None or matched_idxs is None:
            return None

        labels = labels.to(device=scores.device)
        matched_idxs = matched_idxs.to(device=scores.device)
        pos = torch.where(labels > 0)[0]
        if int(pos.numel()) < 2:
            return None

        pair_scores = scores[pos][:, pos]
        same_gt = matched_idxs[pos][:, None].eq(matched_idxs[pos][None, :])
        pair_mask = ~eye[: pos.numel(), : pos.numel()]

        logits = pair_scores[pair_mask]
        targets = same_gt[pair_mask].to(dtype=logits.dtype)
        if logits.numel() == 0:
            return None

        if self.max_edge_pairs is not None and logits.numel() > self.max_edge_pairs:
            keep = torch.randperm(logits.numel(), device=logits.device)[: self.max_edge_pairs]
            logits = logits[keep]
            targets = targets[keep]

        return F.binary_cross_entropy_with_logits(logits, targets)

    @staticmethod
    def _pairwise_geometry(boxes):
        x1, y1, x2, y2 = boxes.unbind(dim=1)
        w = (x2 - x1).clamp(min=1e-6)
        h = (y2 - y1).clamp(min=1e-6)
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h

        dx = (cx[None, :] - cx[:, None]) / w[:, None]
        dy = (cy[None, :] - cy[:, None]) / h[:, None]
        dw = torch.log(w[None, :] / w[:, None])
        dh = torch.log(h[None, :] / h[:, None])
        iou = box_iou(boxes, boxes)

        return torch.stack((dx, dy, dw, dh, iou), dim=-1)
