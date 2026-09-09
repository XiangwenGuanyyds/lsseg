"""Leakage of matched predicted masks into neighboring GT instances."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils


MATCH_IOU_THRESHOLD = 0.5


@dataclass
class InstanceLeakageAccumulator:
    num_gt: int = 0
    num_matched: int = 0
    pred_pixels: int = 0
    fp_pixels: int = 0
    neighbor_pixels: int = 0
    leakage_ratios: list[float] = field(default_factory=list)
    fp_neighbor_shares: list[float] = field(default_factory=list)

    def add(self, pred_area: int, fp_area: int, neighbor_area: int) -> None:
        if not 0 <= neighbor_area <= fp_area <= pred_area:
            raise ValueError("Invalid instance-leakage pixel counts")
        self.num_matched += 1
        self.pred_pixels += pred_area
        self.fp_pixels += fp_area
        self.neighbor_pixels += neighbor_area
        self.leakage_ratios.append(neighbor_area / pred_area if pred_area else 0.0)
        if fp_area:
            self.fp_neighbor_shares.append(neighbor_area / fp_area)

    def result(self, group: str) -> dict[str, float | str]:
        return {
            "group": group,
            "num_gt": float(self.num_gt),
            "num_matched": float(self.num_matched),
            "match_coverage": (
                self.num_matched / self.num_gt if self.num_gt else float("nan")
            ),
            "mean_neighbor_leakage_ratio": (
                mean(self.leakage_ratios)
                if self.leakage_ratios
                else float("nan")
            ),
            "median_neighbor_leakage_ratio": (
                median(self.leakage_ratios)
                if self.leakage_ratios
                else float("nan")
            ),
            "pooled_neighbor_leakage_ratio": (
                self.neighbor_pixels / self.pred_pixels
                if self.pred_pixels
                else float("nan")
            ),
            "mean_neighbor_share_of_instance_fp": (
                mean(self.fp_neighbor_shares)
                if self.fp_neighbor_shares
                else float("nan")
            ),
            "pooled_neighbor_share_of_instance_fp": (
                self.neighbor_pixels / self.fp_pixels
                if self.fp_pixels
                else float("nan")
            ),
        }


def match_predictions(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    iou_threshold: float = MATCH_IOU_THRESHOLD,
) -> list[tuple[int, int, float]]:
    """Score-ordered one-to-one mask matching for one image."""
    if not predictions or not ground_truths:
        return []
    pred_rles = [prediction["rle"] for prediction in predictions]
    gt_rles = [ground_truth["rle"] for ground_truth in ground_truths]
    ious = np.asarray(
        mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles)),
        dtype=np.float64,
    )
    unmatched_gt = np.ones(len(ground_truths), dtype=bool)
    matches: list[tuple[int, int, float]] = []
    for pred_index in range(len(predictions)):
        candidate_indices = np.flatnonzero(unmatched_gt)
        if candidate_indices.size == 0:
            break
        candidate_ious = ious[pred_index, candidate_indices]
        best_offset = int(np.argmax(candidate_ious))
        gt_index = int(candidate_indices[best_offset])
        best_iou = float(candidate_ious[best_offset])
        if best_iou < iou_threshold:
            continue
        unmatched_gt[gt_index] = False
        matches.append((pred_index, gt_index, best_iou))
    return matches


def update_instance_leakage(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    gt_stack: np.ndarray,
    accumulators: dict[str, InstanceLeakageAccumulator],
) -> None:
    """Accumulate leakage statistics from one image."""
    if gt_stack.ndim != 3 or gt_stack.shape[2] != len(ground_truths):
        raise ValueError("GT mask stack does not match the GT instance list")
    gt_coverage_count = gt_stack.sum(axis=2, dtype=np.int16)
    for pred_index, gt_index, _ in match_predictions(predictions, ground_truths):
        pred_mask = mask_utils.decode(predictions[pred_index]["rle"]).astype(bool)
        target_mask = gt_stack[:, :, gt_index]
        other_gt_region = (
            gt_coverage_count - target_mask.astype(np.int16)
        ) > 0
        instance_fp = pred_mask & ~target_mask
        neighbor_pixels = instance_fp & other_gt_region
        values = (
            int(np.count_nonzero(pred_mask)),
            int(np.count_nonzero(instance_fp)),
            int(np.count_nonzero(neighbor_pixels)),
        )
        group = ground_truths[gt_index]["group"]
        accumulators["overall"].add(*values)
        if group in accumulators:
            accumulators[group].add(*values)
