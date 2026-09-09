"""Measure boundary distances and instance leakage from saved predictions."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils

from evaluation.artifact_checks import require_test_artifacts
from .boundary_distance import (
    BoundaryDistanceAccumulator,
    update_boundary_distances,
)
from .instance_leakage import (
    InstanceLeakageAccumulator,
    update_instance_leakage,
)


REQUIRED_ARTIFACTS = {
    "predictions": ("predictions.json",),
    "ground_truth": ("ground_truth/overall.json",),
    "occluded_ground_truth": ("ground_truth/occluded.json",),
    "not_occluded_ground_truth": ("ground_truth/not_occluded.json",),
}
GROUPS = ("overall", "occluded", "not_occluded")


def _annotation_rle(annotation: dict[str, Any], height: int, width: int):
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        return mask_utils.merge(
            mask_utils.frPyObjects(segmentation, height, width)
        )
    if isinstance(segmentation, dict) and isinstance(
        segmentation.get("counts"), list
    ):
        return mask_utils.frPyObjects(segmentation, height, width)
    return segmentation


def _load_ground_truth(
    overall_path: Path,
    occluded_path: Path,
    not_occluded_path: Path,
) -> tuple[dict[int, tuple[int, int]], dict[int, list[dict[str, Any]]]]:
    with overall_path.open() as handle:
        overall = json.load(handle)
    with occluded_path.open() as handle:
        occluded = json.load(handle)
    with not_occluded_path.open() as handle:
        not_occluded = json.load(handle)

    image_sizes = {
        int(image["id"]): (int(image["height"]), int(image["width"]))
        for image in overall["images"]
    }
    occluded_ids = {
        int(annotation["id"])
        for annotation in occluded["annotations"]
        if int(annotation.get("iscrowd", 0)) == 0
    }
    not_occluded_ids = {
        int(annotation["id"])
        for annotation in not_occluded["annotations"]
        if int(annotation.get("iscrowd", 0)) == 0
    }
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in overall["annotations"]:
        if int(annotation.get("iscrowd", 0)) != 0:
            continue
        image_id = int(annotation["image_id"])
        height, width = image_sizes[image_id]
        annotation_id = int(annotation["id"])
        by_image[image_id].append(
            {
                "id": annotation_id,
                "rle": _annotation_rle(annotation, height, width),
                "group": (
                    "occluded" if annotation_id in occluded_ids
                    else "not_occluded" if annotation_id in not_occluded_ids
                    else None
                ),
            }
        )
    return image_sizes, by_image


def _load_predictions(path: Path) -> dict[int, list[dict[str, Any]]]:
    with path.open() as handle:
        predictions = json.load(handle)
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, prediction in enumerate(predictions):
        if int(prediction["category_id"]) != 1:
            continue
        by_image[int(prediction["image_id"])].append(
            {
                "candidate_index": index,
                "score": float(prediction["score"]),
                "rle": prediction["segmentation"],
            }
        )
    for image_predictions in by_image.values():
        image_predictions.sort(
            key=lambda item: (-item["score"], item["candidate_index"])
        )
    return by_image


def _decode_stack(rles: list, height: int, width: int) -> np.ndarray:
    if not rles:
        return np.zeros((height, width, 0), dtype=bool)
    decoded = mask_utils.decode(rles).astype(bool)
    if decoded.ndim == 2:
        decoded = decoded[:, :, None]
    return decoded


def _decode_union(rles: list, height: int, width: int) -> np.ndarray:
    if not rles:
        return np.zeros((height, width), dtype=bool)
    return mask_utils.decode(mask_utils.merge(rles)).astype(bool)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        f"{value:.8f}"
                        if isinstance(value, float) and math.isfinite(value)
                        else value
                    )
                    for key, value in row.items()
                }
            )


def evaluate_mask_errors(
    result_dir: Path,
    experiment: str,
    output_dir: Path | None = None,
) -> dict:
    """Compute boundary-distance and instance-leakage statistics for one test result."""
    result_dir = Path(result_dir)
    artifacts = require_test_artifacts(
        result_dir,
        experiment,
        REQUIRED_ARTIFACTS,
    )
    image_sizes, gt_by_image = _load_ground_truth(
        artifacts["ground_truth"],
        artifacts["occluded_ground_truth"],
        artifacts["not_occluded_ground_truth"],
    )
    predictions_by_image = _load_predictions(artifacts["predictions"])

    fp_accumulator = BoundaryDistanceAccumulator()
    fn_accumulator = BoundaryDistanceAccumulator()
    leakage_accumulators = {
        group: InstanceLeakageAccumulator() for group in GROUPS
    }
    group_counts = Counter(
        ground_truth["group"]
        for ground_truths in gt_by_image.values()
        for ground_truth in ground_truths
    )
    leakage_accumulators["overall"].num_gt = sum(group_counts.values())
    leakage_accumulators["occluded"].num_gt = group_counts["occluded"]
    leakage_accumulators["not_occluded"].num_gt = group_counts["not_occluded"]

    for image_id, (height, width) in sorted(image_sizes.items()):
        ground_truths = gt_by_image.get(image_id, [])
        predictions = predictions_by_image.get(image_id, [])
        gt_stack = _decode_stack(
            [item["rle"] for item in ground_truths], height, width
        )
        gt_union = np.any(gt_stack, axis=2)
        pred_union = _decode_union(
            [item["rle"] for item in predictions], height, width
        )
        update_boundary_distances(
            gt_union,
            pred_union,
            fp_accumulator,
            fn_accumulator,
        )
        update_instance_leakage(
            predictions,
            ground_truths,
            gt_stack,
            leakage_accumulators,
        )

    boundary_rows = [
        fp_accumulator.result("FP"),
        fn_accumulator.result("FN"),
    ]
    leakage_rows = [
        leakage_accumulators[group].result(group) for group in GROUPS
    ]
    output_dir = Path(output_dir) if output_dir is not None else result_dir.parent / "analysis" / "mask_errors"
    boundary_path = output_dir / "boundary_distance.csv"
    leakage_path = output_dir / "instance_leakage.csv"
    _write_csv(boundary_path, boundary_rows)
    _write_csv(leakage_path, leakage_rows)
    return {
        "artifacts": artifacts,
        "boundary_distance": boundary_rows,
        "instance_leakage": leakage_rows,
        "outputs": {
            "boundary_distance": boundary_path,
            "instance_leakage": leakage_path,
        },
    }


# Compatibility alias for existing external callers.
evaluate_result_directory = evaluate_mask_errors


def print_summary(experiment: str, result: dict) -> None:
    print(f"Mask error analysis: {experiment}")
    print("\nBoundary distance")
    print(
        f"{'type':>6s}  {'pixels':>12s}  {'mean px':>10s}  "
        f"{'median':>10s}  {'within 5 px':>12s}"
    )
    for row in result["boundary_distance"]:
        print(
            f"{row['error_type']:>6s}  {int(row['num_pixels']):>12d}  "
            f"{float(row['mean_distance_px']):>10.4f}  "
            f"{float(row['median_distance_px']):>10.4f}  "
            f"{float(row['fraction_within_5px']):>12.4f}"
        )

    print("\nInstance leakage")
    print(
        f"{'group':16s}  {'match coverage':>14s}  "
        f"{'pooled leakage':>15s}  {'FP neighbor share':>17s}"
    )
    for row in result["instance_leakage"]:
        print(
            f"{row['group']:16s}  {float(row['match_coverage']):>14.4f}  "
            f"{float(row['pooled_neighbor_leakage_ratio']):>15.4f}  "
            f"{float(row['pooled_neighbor_share_of_instance_fp']):>17.4f}"
        )
    for name, path in result["outputs"].items():
        print(f"wrote {name}: {path}")
