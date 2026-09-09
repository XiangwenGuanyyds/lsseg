"""Distance of foreground FP/FN pixels to the ground-truth mask boundary."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


BOUNDARY_THRESHOLDS = (1.0, 2.0, 3.0, 5.0)


@dataclass
class BoundaryDistanceAccumulator:
    total_count: int = 0
    distance_sum: float = 0.0
    distance_counts: Counter[float] = field(default_factory=Counter)
    near_counts: dict[float, int] = field(
        default_factory=lambda: {
            threshold: 0 for threshold in BOUNDARY_THRESHOLDS
        }
    )

    def update(self, distances: np.ndarray) -> None:
        distances = np.asarray(distances, dtype=np.float64)
        if distances.size == 0:
            return
        self.total_count += int(distances.size)
        self.distance_sum += float(distances.sum())
        values, counts = np.unique(distances, return_counts=True)
        for value, count in zip(values, counts):
            self.distance_counts[float(value)] += int(count)
        for threshold in BOUNDARY_THRESHOLDS:
            self.near_counts[threshold] += int(
                np.count_nonzero(distances <= threshold)
            )

    def _weighted_quantile(self, quantile: float) -> float:
        if self.total_count == 0:
            return float("nan")
        target = quantile * self.total_count
        cumulative = 0
        for value, count in sorted(self.distance_counts.items()):
            cumulative += count
            if cumulative >= target:
                return value
        return max(self.distance_counts)

    def result(self, error_type: str) -> dict[str, float | str]:
        row: dict[str, float | str] = {
            "error_type": error_type,
            "num_pixels": float(self.total_count),
            "mean_distance_px": (
                self.distance_sum / self.total_count
                if self.total_count
                else float("nan")
            ),
            "median_distance_px": self._weighted_quantile(0.5),
            "p90_distance_px": self._weighted_quantile(0.9),
        }
        for threshold in BOUNDARY_THRESHOLDS:
            row[f"fraction_within_{int(threshold)}px"] = (
                self.near_counts[threshold] / self.total_count
                if self.total_count
                else float("nan")
            )
        return row


def update_boundary_distances(
    gt_union: np.ndarray,
    pred_union: np.ndarray,
    fp_accumulator: BoundaryDistanceAccumulator,
    fn_accumulator: BoundaryDistanceAccumulator,
) -> None:
    """Accumulate FP/FN distances for one image."""
    gt_union = np.asarray(gt_union, dtype=bool)
    pred_union = np.asarray(pred_union, dtype=bool)
    if gt_union.shape != pred_union.shape:
        raise ValueError("GT and prediction union masks must have equal shapes")

    gt_boundary = gt_union & ~binary_erosion(
        gt_union,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    fp_mask = pred_union & ~gt_union
    fn_mask = gt_union & ~pred_union
    if not bool(gt_boundary.any()):
        if bool(fp_mask.any()) or bool(fn_mask.any()):
            raise ValueError("Cannot measure boundary distance without a GT boundary")
        return

    distance_to_boundary = distance_transform_edt(~gt_boundary)
    fp_accumulator.update(distance_to_boundary[fp_mask])
    fn_accumulator.update(distance_to_boundary[fn_mask])
