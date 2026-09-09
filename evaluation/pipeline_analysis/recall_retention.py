"""Compare detection recall before and after post-processing.

At each IoU threshold, match boxes to same-class GT in descending score order.
Matching is one-to-one and runs separately for the pre- and post-processing sets.
"""

import csv
import math
import os

import torch
from torchvision.ops import box_iou


_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
_GROUPS = ("overall", "occluded", "not_occluded")


def _matched_gt_mask(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, threshold):
    """Return which GT boxes are matched by score-ordered one-to-one matching."""
    matched = torch.zeros(
        gt_boxes.shape[0], dtype=torch.bool, device=gt_boxes.device
    )
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return matched

    # Keep input order when scores are equal.
    order = torch.argsort(pred_scores, descending=True, stable=True)
    for pred_index in order.tolist():
        same_class = (gt_labels == pred_labels[pred_index]) & (~matched)
        if not bool(same_class.any()):
            continue

        gt_indices = torch.where(same_class)[0]
        ious = box_iou(
            pred_boxes[pred_index:pred_index + 1], gt_boxes[gt_indices]
        )[0]
        best_pos = int(torch.argmax(ious).item())
        if float(ious[best_pos]) >= threshold:
            matched[gt_indices[best_pos]] = True

    return matched


def accumulate_postprocessing_recall(pre_batches, post_batches, targets, state):
    if state is None or targets is None:
        return
    if not state.get("diagnostics", {}).get("postprocessing_recall", False):
        return

    accum = state.setdefault("postprocessing_recall_accumulator", _empty_accum())
    if len(pre_batches) != len(post_batches) or len(pre_batches) != len(targets):
        raise ValueError("pre/post candidate batches and targets must have equal length")

    for pre, post, target in zip(pre_batches, post_batches, targets):
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]
        is_occluded = target.get(
            "is_occluded",
            torch.zeros(gt_boxes.shape[0], dtype=torch.bool, device=gt_boxes.device),
        )
        has_label = target.get(
            "has_occlusion_label",
            torch.ones(gt_boxes.shape[0], dtype=torch.bool, device=gt_boxes.device),
        )
        if gt_boxes.numel() == 0:
            continue

        for threshold in _THRESHOLDS:
            pre_matched = _matched_gt_mask(
                pre["boxes"], pre["scores"], pre["labels"],
                gt_boxes, gt_labels, threshold,
            )
            post_matched = _matched_gt_mask(
                post["boxes"], post["scores"], post["labels"],
                gt_boxes, gt_labels, threshold,
            )

            group_masks = {
                "overall": torch.ones_like(is_occluded, dtype=torch.bool),
                "occluded": is_occluded.bool() & has_label.bool(),
                "not_occluded": (~is_occluded.bool()) & has_label.bool(),
            }
            for group, group_mask in group_masks.items():
                item = accum[group][threshold]
                item["n_gt"] += int(group_mask.sum().item())
                item["pre_matched"] += int((pre_matched & group_mask).sum().item())
                item["post_matched"] += int((post_matched & group_mask).sum().item())


def write_postprocessing_recall(state, epoch):
    if state is None:
        return
    accum = state.get("postprocessing_recall_accumulator")
    log_dir = state.get("pipeline_analysis_dir", state.get("log_dir"))
    if accum is None or log_dir is None:
        return

    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "postprocessing_recall.csv")
    fields = [
        "epoch", "iou_threshold", "group", "num_gt",
        "num_pre_matched_gt", "num_post_matched_gt",
        "recall_pre", "recall_post", "recall_retention",
    ]
    rows = []
    for threshold in _THRESHOLDS:
        for group in _GROUPS:
            item = accum[group][threshold]
            n_gt = item["n_gt"]
            pre_matched = item["pre_matched"]
            post_matched = item["post_matched"]
            pre_recall = pre_matched / n_gt if n_gt else math.nan
            post_recall = post_matched / n_gt if n_gt else math.nan
            rows.append({
                "epoch": epoch,
                "iou_threshold": threshold,
                "group": group,
                "num_gt": n_gt,
                "num_pre_matched_gt": pre_matched,
                "num_post_matched_gt": post_matched,
                "recall_pre": pre_recall,
                "recall_post": post_recall,
                "recall_retention": (
                    post_recall / pre_recall if pre_recall > 0 else math.nan
                ),
            })

    write_header = True
    file_mode = "w"
    if os.path.exists(path):
        with open(path, "r", newline="") as existing:
            existing_header = next(csv.reader(existing), None)
        if existing_header == fields:
            write_header = False
            file_mode = "a"

    with open(path, file_mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _empty_accum():
    return {
        group: {
            threshold: {
                "n_gt": 0,
                "pre_matched": 0,
                "post_matched": 0,
            }
            for threshold in _THRESHOLDS
        }
        for group in _GROUPS
    }
