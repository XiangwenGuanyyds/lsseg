"""Overall foreground pixel metrics computed from saved test predictions."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pycocotools import mask as mask_utils


OUTPUT_FIELDS = (
    "experiment",
    "group",
    "score_threshold",
    "ground_truth_path",
    "ground_truth_sha256",
    "global_foreground_iou",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "tp_pixels",
    "fp_pixels",
    "fn_pixels",
    "n_pred_instances",
    "n_gt_instances",
)


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


def load_ground_truth(path: Path) -> dict[int, list[dict[str, Any]]]:
    """Load eligible GT instance masks grouped by image ID."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ground-truth file not found: {path}")
    with path.open() as handle:
        coco = json.load(handle)

    image_sizes = {
        int(image["id"]): (int(image["height"]), int(image["width"]))
        for image in coco["images"]
    }
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        if int(annotation.get("iscrowd", 0)) != 0:
            continue
        image_id = int(annotation["image_id"])
        height, width = image_sizes[image_id]
        rle = _annotation_rle(annotation, height, width)
        # print("[debug] image", image_id, "H W:", height, width, "mask size:", rle["size"])
        by_image[image_id].append({"id": annotation["id"], "rle": rle})
    return by_image


def load_predictions(path: Path) -> dict[int, list[dict[str, Any]]]:
    """Load predicted instance masks grouped and score-sorted by image ID."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"prediction file not found: {path}")
    with path.open() as handle:
        predictions = json.load(handle)
    # print("[debug] predictions loaded:", len(predictions), "from", path)

    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        by_image[int(prediction["image_id"])].append(
            {
                "rle": prediction["segmentation"],
                "score": float(prediction["score"]),
            }
        )
    for image_predictions in by_image.values():
        image_predictions.sort(key=lambda item: item["score"], reverse=True)
    return by_image


def compute_overall_pixel_metrics(
    ground_truth: dict[int, list[dict[str, Any]]],
    predictions: dict[int, list[dict[str, Any]]],
    score_threshold: float = 0.0,
) -> dict[str, float | int]:
    """Compute dataset-level foreground metrics from per-image mask unions."""
    tp_pixels = 0.0
    fp_pixels = 0.0
    fn_pixels = 0.0
    n_pred_instances = 0
    n_gt_instances = 0

    # Prediction-only images also need counting.
    for image_id in set(ground_truth) | set(predictions):
        image_gt = ground_truth.get(image_id, [])
        image_predictions = [
            prediction
            for prediction in predictions.get(image_id, [])
            if prediction["score"] >= score_threshold
        ]
        n_gt_instances += len(image_gt)
        n_pred_instances += len(image_predictions)
        # print("[debug] image", image_id, "GT count", len(image_gt), "pred kept", len(image_predictions), "threshold", score_threshold)

        gt_rles = [item["rle"] for item in image_gt]
        pred_rles = [item["rle"] for item in image_predictions]
        # Merge first, overlapping pixels count once.
        gt_union = mask_utils.merge(gt_rles) if gt_rles else None
        pred_union = mask_utils.merge(pred_rles) if pred_rles else None
        # print("[debug] union mask sizes H W:", gt_union["size"] if gt_union is not None else None, pred_union["size"] if pred_union is not None else None)
        gt_area = float(mask_utils.area(gt_union)) if gt_union is not None else 0.0
        pred_area = (
            float(mask_utils.area(pred_union)) if pred_union is not None else 0.0
        )
        if gt_union is not None and pred_union is not None:
            iou = float(mask_utils.iou([pred_union], [gt_union], [0])[0, 0])
            # Get intersection area back from IoU.
            intersection = iou * (pred_area + gt_area) / (1.0 + iou)
        else:
            intersection = 0.0
        # print("[debug] areas GT / pred / intersection", gt_area, pred_area, intersection)

        tp_pixels += intersection
        fp_pixels += pred_area - intersection
        fn_pixels += gt_area - intersection

    # Ratios use pixel totals from all images.
    # print("[debug] total TP FP FN:", tp_pixels, fp_pixels, fn_pixels)
    precision = (
        tp_pixels / (tp_pixels + fp_pixels)
        if tp_pixels + fp_pixels > 0
        else 0.0
    )
    recall = (
        tp_pixels / (tp_pixels + fn_pixels)
        if tp_pixels + fn_pixels > 0
        else 0.0
    )
    pixel_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    foreground_iou = (
        tp_pixels / (tp_pixels + fp_pixels + fn_pixels)
        if tp_pixels + fp_pixels + fn_pixels > 0
        else 0.0
    )
    # print("[debug] IoU", foreground_iou, "precision", precision, "recall", recall, "F1", pixel_f1)
    return {
        "global_foreground_iou": foreground_iou,
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": pixel_f1,
        "tp_pixels": tp_pixels,
        "fp_pixels": fp_pixels,
        "fn_pixels": fn_pixels,
        "n_pred_instances": n_pred_instances,
        "n_gt_instances": n_gt_instances,
    }


def evaluate_foreground_metrics(
    result_dir: Path,
    score_threshold: float = 0.0,
    ground_truth_path: Path | None = None,
) -> dict[str, float | int]:
    """Evaluate one saved test result directory."""
    result_dir = Path(result_dir)
    ground_truth_path = Path(
        ground_truth_path or result_dir / "ground_truth" / "overall.json"
    )
    ground_truth = load_ground_truth(ground_truth_path)
    predictions = load_predictions(result_dir / "predictions.json")
    return compute_overall_pixel_metrics(
        ground_truth,
        predictions,
        score_threshold=score_threshold,
    )


# Compatibility alias for existing external callers.
evaluate_result_directory = evaluate_foreground_metrics


def write_overall_pixel_metrics(
    output_dir: Path,
    experiment: str,
    metrics: dict[str, float | int],
    score_threshold: float = 0.0,
    ground_truth_path: Path | None = None,
) -> Path:
    """Write overall pixel metrics and evaluation settings to overall.csv."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "overall.csv"
    resolved_ground_truth = (
        Path(ground_truth_path).resolve() if ground_truth_path is not None else None
    )
    ground_truth_sha256 = ""
    if resolved_ground_truth is not None:
        digest = hashlib.sha256()
        with resolved_ground_truth.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        ground_truth_sha256 = digest.hexdigest()
    row = {
        "experiment": experiment,
        "group": "overall",
        "score_threshold": score_threshold,
        "ground_truth_path": str(resolved_ground_truth or ""),
        "ground_truth_sha256": ground_truth_sha256,
        **metrics,
    }
    # Rerun will overwrite this CSV.
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    # print("[debug] result saved", output_path)
    return output_path
