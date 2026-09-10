"""Box NMS methods selected through cfg.MODEL["box_nms"].

Inputs are boxes, scores, labels, and NMS parameters.
Returns retained input indices and final scores, ordered by descending score.
"""

import torch
from torchvision.ops import box_iou, boxes as box_ops

from utils import BOX_NMS


@BOX_NMS.register_module(name="standard_nms")
def standard_nms(boxes, scores, labels, nms_thresh=0.5, **_):
    """Standard per-class NMS using torchvision."""
    keep = box_ops.batched_nms(boxes, scores, labels, nms_thresh)
    return keep, scores[keep]


@BOX_NMS.register_module(name="gaussian_soft_nms")
def gaussian_soft_nms(boxes, scores, labels,
             sigma=0.5, score_thresh_keep=1e-3, **_):
    """Per-class Gaussian Soft-NMS (Bodla et al. 2017).

    Update remaining same-class scores using IoU with the selected box:
        new_score = old_score * exp(-iou^2 / sigma)
    Keep boxes with scores above score_thresh_keep.
    """
    if sigma <= 0:
        raise ValueError("Gaussian Soft-NMS sigma must be positive")
    if score_thresh_keep < 0:
        raise ValueError("Soft-NMS score_thresh_keep must be non-negative")

    if boxes.numel() == 0:
        return (torch.empty(0, dtype=torch.long, device=boxes.device),
                torch.empty(0, dtype=scores.dtype, device=boxes.device))

    # copy scores first
    work_scores = scores.clone()
    # print("[debug]", boxes.shape, scores.shape)
    # [debug] torch.Size([3, 4]) torch.Size([3])
    keep_indices = []
    keep_scores  = []
    N = boxes.shape[0]
    # IoU calculated once here
    pairwise_iou = box_iou(boxes, boxes)

    for _ in range(N):
        if work_scores.max() <= score_thresh_keep:
            break
        max_idx = int(work_scores.argmax().item())
        max_score = float(work_scores[max_idx].item())
        keep_indices.append(max_idx)
        keep_scores.append(max_score)
        work_scores[max_idx] = 0.0  # already picked

        same_label = (labels == labels[max_idx]) & (work_scores > 0)
        if not same_label.any():
            continue
        idx_same = torch.where(same_label)[0]
        ious = pairwise_iou[max_idx, idx_same]
        decay = torch.exp(-(ious ** 2) / sigma)
        work_scores[idx_same] *= decay

    keep = torch.tensor(keep_indices, dtype=torch.long, device=boxes.device)
    kept_scores = torch.tensor(keep_scores, dtype=scores.dtype, device=boxes.device)
    return keep, kept_scores


@BOX_NMS.register_module(name="thresholded_gaussian_soft_nms")
def thresholded_gaussian_soft_nms(
    boxes,
    scores,
    labels,
    nms_thresh=0.5,
    sigma=0.5,
    score_thresh_keep=1e-3,
    **_,
):
    """Gaussian Soft-NMS with an IoU threshold.

    Apply Gaussian score decay to remaining same-class boxes whose IoU with
    the selected box exceeds nms_thresh.
    """
    if not 0 <= nms_thresh <= 1:
        raise ValueError("Thresholded Gaussian NMS threshold must be in [0, 1]")
    if sigma <= 0:
        raise ValueError("Thresholded Gaussian NMS sigma must be positive")
    if score_thresh_keep < 0:
        raise ValueError("Soft-NMS score_thresh_keep must be non-negative")

    if boxes.numel() == 0:
        return (
            torch.empty(0, dtype=torch.long, device=boxes.device),
            torch.empty(0, dtype=scores.dtype, device=boxes.device),
        )

    work_scores = scores.clone()
    keep_indices = []
    keep_scores = []
    pairwise_iou = box_iou(boxes, boxes)

    for _ in range(boxes.shape[0]):
        if work_scores.max() <= score_thresh_keep:
            break
        # scores changed, pick max again
        max_idx = int(work_scores.argmax().item())
        keep_indices.append(max_idx)
        keep_scores.append(float(work_scores[max_idx].item()))
        work_scores[max_idx] = 0.0

        same_label = (labels == labels[max_idx]) & (work_scores > 0)
        if not same_label.any():
            continue
        same_indices = torch.where(same_label)[0]
        ious = pairwise_iou[max_idx, same_indices]
        # IoU above threshold
        decay_mask = ious > nms_thresh
        if decay_mask.any():
            decay_indices = same_indices[decay_mask]
            work_scores[decay_indices] *= torch.exp(
                -(ious[decay_mask] ** 2) / sigma
            )

    keep = torch.tensor(keep_indices, dtype=torch.long, device=boxes.device)
    kept_scores = torch.tensor(
        keep_scores, dtype=scores.dtype, device=boxes.device
    )
    # print("[debug] keep", keep, kept_scores)
    # [debug] keep tensor([0, 2, 1]) tensor([0.9000, 0.7000, 0.2097])
    return keep, kept_scores
