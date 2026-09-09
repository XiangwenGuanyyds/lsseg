"""Measure how many GT boxes are covered by RPN proposals.

A GT box is covered when its maximum IoU with proposals in the same image
reaches the threshold. Sum GT and covered counts across batches, then divide
the totals for each threshold and group. Thresholds are 0.3, 0.5 and 0.7.

Groups use target["is_occluded"] and target["has_occlusion_label"].
Instances without an agreed label contribute only to the overall group.
"""

import csv
import os

import torch
from torchvision.ops import box_iou


_THRESHOLDS = (0.3, 0.5, 0.7)


def accumulate_rpn_coverage(proposals, targets, state):
    """Accumulate batch coverage when RPN diagnostics are enabled."""
    if state is None or targets is None:
        return
    if not state.get("diagnostics", {}).get("rpn_coverage", False):
        return

    accum = state.setdefault("rpn_coverage_accumulator", _empty_accum())

    for img_props, tgt in zip(proposals, targets):
        gt_boxes = tgt["boxes"]
        if gt_boxes.shape[0] == 0:
            continue
        is_occ = tgt.get(
            "is_occluded",
            torch.zeros(gt_boxes.shape[0], dtype=torch.bool, device=gt_boxes.device),
        )
        has_label = tgt.get(
            "has_occlusion_label",
            torch.ones(gt_boxes.shape[0], dtype=torch.bool, device=gt_boxes.device),
        )

        if img_props.shape[0] == 0:
            max_iou = torch.zeros(gt_boxes.shape[0], device=gt_boxes.device)
        else:
            max_iou = box_iou(gt_boxes, img_props).max(dim=1).values

        for t in _THRESHOLDS:
            covered = max_iou >= t
            group_masks = {
                "all": torch.ones_like(is_occ, dtype=torch.bool),
                "occluded": is_occ & has_label,
                "not_occluded": (~is_occ) & has_label,
            }
            for grp_name, mask in group_masks.items():
                accum[t][grp_name]["n_gt"]      += int(mask.sum())
                accum[t][grp_name]["n_covered"] += int((covered & mask).sum())


def write_rpn_coverage(state, epoch):
    """Append one row per threshold and group to rpn_coverage.csv."""
    accum = state.get("rpn_coverage_accumulator")
    if accum is None:
        return
    log_dir = state.get("pipeline_analysis_dir", state.get("log_dir"))
    if log_dir is None:
        return

    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "rpn_coverage.csv")

    rows = []
    for t in sorted(accum.keys()):
        for grp_name in ("occluded", "not_occluded", "all"):
            d = accum[t][grp_name]
            rows.append((epoch, t, grp_name, d["n_gt"], d["n_covered"], _rate(d)))

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["epoch", "threshold", "group", "n_gt", "n_covered", "coverage_rate"])
        for row in rows:
            w.writerow(row)

def _empty_accum():
    return {t: {"all":          {"n_gt": 0, "n_covered": 0},
                "occluded":     {"n_gt": 0, "n_covered": 0},
                "not_occluded": {"n_gt": 0, "n_covered": 0}}
            for t in _THRESHOLDS}


def _rate(d):
    return d["n_covered"] / d["n_gt"] if d["n_gt"] else 0.0
