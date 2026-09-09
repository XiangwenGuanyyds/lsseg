"""Generate bbox and segmentation COCO AP reports from saved test outputs."""

import contextlib
import csv
import io
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


GT_FILES = (
    ("overall", "ground_truth/overall.json"),
    ("occluded", "ground_truth/occluded.json"),
    ("not_occluded", "ground_truth/not_occluded.json"),
)

REPORT_FIELDS = ("exp_name", "group", "iou_type", "AP", "AP50", "AP75")


def _load_coco(path):
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(path))


def _evaluate(gt_path, predictions, iou_type):
    if not predictions:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}

    gt = _load_coco(gt_path)
    # print("[debug] gt:", gt_path, "images:", len(gt.imgs))
    # print("[debug] iou type:", iou_type, "prediction count:", len(predictions))
    # COCOeval mutates its inputs. Copy the predictions so bbox and segm
    # evaluations, as well as the three groups, remain independent.
    predictions = json.loads(json.dumps(predictions))
    with contextlib.redirect_stdout(io.StringIO()):
        pred_coco = gt.loadRes(predictions)
        evaluator = COCOeval(gt, pred_coco, iou_type)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    # print("[debug] COCO result annotations", len(pred_coco.anns))
    # print("[debug] raw evaluator.stats:", evaluator.stats)
    metrics = {
        "AP": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
    }
    # print("[debug] metrics", iou_type, metrics)
    return metrics


def build_coco_ap_rows(exp_dir):
    """Evaluate all available GT groups for bbox and segmentation AP."""
    exp_dir = Path(exp_dir)
    pred_path = exp_dir / "predictions.json"
    if not pred_path.exists():
        raise FileNotFoundError(f"missing test predictions: {pred_path}")

    with pred_path.open() as f:
        predictions = json.load(f)
    # print("[debug] loaded", len(predictions), "predictions from", pred_path)

    rows = []
    for group, filename in GT_FILES:
        gt_path = exp_dir / filename
        if not gt_path.exists():
            continue
        for iou_type in ("bbox", "segm"):
            # print("[debug] evaluating group/type:", group, iou_type)
            metrics = _evaluate(gt_path, predictions, iou_type)
            rows.append({
                "exp_name": exp_dir.parent.name,
                "group": group,
                "iou_type": iou_type,
                "AP": metrics["AP"],
                "AP50": metrics["AP50"],
                "AP75": metrics["AP75"],
            })
    # print("[debug] report rows:", rows)
    return rows


def write_coco_ap_report(exp_dir, metrics_dir=None):
    """Write ``coco_ap_report.csv`` and return its path and rows."""
    exp_dir = Path(exp_dir)
    rows = build_coco_ap_rows(exp_dir)
    metrics_dir = Path(metrics_dir) if metrics_dir is not None else exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / "coco_ap_report.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    # print("[debug] wrote COCO AP report to", out_path)
    return out_path, rows
