"""Compute mask AP, AP50 and AP75 with pycocotools.

For grouped evaluation, GT outside the selected group is marked iscrowd=1.
These annotations are ignored by COCOeval when computing the group's AP.
"""

import contextlib
import csv
import io
import json
import os

import numpy as np
import pycocotools.mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class MaskAPEvaluator:

    def __init__(self, ann_file, eval_class_ids, occluded_ann_ids=None,
                 not_occluded_ann_ids=None, category_remap=None, debug=False):
        """Set up overall AP and optional occlusion groups.

        With both ID sets supplied, GT outside either group stays in overall AP
        only. If not_occluded_ann_ids is omitted, use all GT IDs outside the
        occluded set for the not-occluded group.
        """
        self.eval_class_ids = list(eval_class_ids)
        self.debug = debug

        # Hide pycocotools loading and indexing messages.
        with contextlib.redirect_stdout(io.StringIO()):
            self._gt_full    = self._load_gt(ann_file, category_remap)
            self._gt_subsets = self._build_subsets(
                occluded_ann_ids, not_occluded_ann_ids
            )

        self._predictions = []

    def _load_gt(self, ann_file, category_remap):
        if category_remap is None:
            return COCO(ann_file)

        with open(ann_file) as f:
            raw = json.load(f)

        new_anns = []
        for ann in raw.get("annotations", []):
            src = ann.get("category_id")
            if src not in category_remap:
                continue
            new_ann = dict(ann)
            new_ann["category_id"] = category_remap[src]
            new_anns.append(new_ann)
        raw["annotations"] = new_anns

        old_cats = {c["id"]: c for c in raw.get("categories", [])}
        new_cats = []
        for src, dst in category_remap.items():
            if src not in old_cats:
                continue
            cat = dict(old_cats[src])
            cat["id"] = dst
            new_cats.append(cat)
        raw["categories"] = new_cats

        coco = COCO()
        coco.dataset = raw
        coco.createIndex()
        return coco

    def _build_subsets(self, occluded_ann_ids, not_occluded_ann_ids=None):
        """Build COCO subsets and validate the supplied group IDs.

        With no occluded IDs and no explicit not-occluded set, return no groups.
        """
        if not occluded_ann_ids and not_occluded_ann_ids is None:
            return {}

        occluded_ids = {int(x) for x in occluded_ann_ids}
        gt_ids = {a["id"] for a in self._gt_full.dataset["annotations"]}
        if not_occluded_ann_ids is None:
            not_occluded_ids = gt_ids - occluded_ids
        else:
            not_occluded_ids = {int(x) for x in not_occluded_ann_ids}
            overlap = occluded_ids & not_occluded_ids
            unknown = (occluded_ids | not_occluded_ids) - gt_ids
            if overlap:
                raise ValueError(
                    f"Occlusion groups overlap for {len(overlap)} annotation ids"
                )
            if unknown:
                raise ValueError(
                    f"Occlusion groups contain {len(unknown)} unknown annotation ids"
                )
        return {
            "occluded":     self._mark_others_iscrowd(occluded_ids),
            "not_occluded": self._mark_others_iscrowd(not_occluded_ids),
        }

    def _mark_others_iscrowd(self, wanted_ann_ids):
        new_anns = []
        for a in self._gt_full.dataset["annotations"]:
            new_ann = dict(a)
            if a["id"] not in wanted_ann_ids:
                new_ann["iscrowd"] = 1
            new_anns.append(new_ann)
        subset = dict(self._gt_full.dataset)
        subset["annotations"] = new_anns
        coco = COCO()
        coco.dataset = subset
        coco.createIndex()
        return coco

    def update_batch(self, detections, image_ids):
        """detections: list of {boxes, labels, scores, masks}, one per image,
        masks already at original-image resolution and float in [0,1].
        image_ids:  list of int, parallel to detections."""
        for det, img_id in zip(detections, image_ids):
            boxes  = det["boxes"]
            labels = det["labels"]
            scores = det["scores"]
            masks  = det["masks"]
            if masks.numel() == 0:
                continue
            for i in range(labels.shape[0]):
                cls_id = int(labels[i].item())
                if cls_id not in self.eval_class_ids:
                    continue
                m = masks[i]
                if m.dim() == 3:
                    m = m[0]
                binary = (m.cpu().numpy() > 0.5).astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(binary))
                if isinstance(rle, list):
                    rle = rle[0]
                counts = rle["counts"]
                if isinstance(counts, bytes):
                    counts = counts.decode("ascii")
                x1, y1, x2, y2 = [float(v) for v in boxes[i].tolist()]
                self._predictions.append({
                    "image_id":     int(img_id),
                    "category_id":  cls_id,
                    "bbox":         [x1, y1, x2 - x1, y2 - y1],
                    "segmentation": {"size": list(rle["size"]), "counts": counts},
                    "score":        float(scores[i].item()),
                })

    def dump(self, out_dir, ground_truth_dir=None):
        """Write predictions and remapped ground truth as JSON files.

        predictions.json contains COCO detections with boxes and RLE masks.
        The GT directory contains overall.json and one JSON per occlusion group.
        GT outside each group keeps iscrowd=1 in that group's file.

        The default GT directory is out_dir/ground_truth.
        """
        ground_truth_dir = ground_truth_dir or os.path.join(out_dir, "ground_truth")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(ground_truth_dir, exist_ok=True)
        with open(os.path.join(out_dir, "predictions.json"), "w") as f:
            json.dump(self._predictions, f)
        with open(os.path.join(ground_truth_dir, "overall.json"), "w") as f:
            json.dump(self._gt_full.dataset, f)
        for grp, coco in self._gt_subsets.items():
            with open(os.path.join(ground_truth_dir, f"{grp}.json"), "w") as f:
                json.dump(coco.dataset, f)

    def compute(self):
        if self.debug:
            self._debug_dump()
        out = {"overall": self._eval(self._gt_full)}
        for grp, subset_coco in self._gt_subsets.items():
            out[grp] = self._eval(subset_coco)
        return out

    def _debug_dump(self):
        """Print GT counts, ignored counts and a sample prediction."""
        def _split(anns):
            elig = sum(1 for a in anns if a.get("iscrowd", 0) == 0)
            ign  = sum(1 for a in anns if a.get("iscrowd", 0) == 1)
            return elig, ign

        print("[DEBUG/mask_ap] ────── COCOeval input check ──────")
        print(f"  eval_class_ids : {self.eval_class_ids}")
        print(f"  GT categories  : {self._gt_full.dataset['categories']}")
        n_imgs = len(self._gt_full.dataset.get("images", []))
        e, i = _split(self._gt_full.dataset["annotations"])
        print(f"  GT overall     : {e} eligible + {i} ignored across {n_imgs} images")
        for grp, subset_coco in self._gt_subsets.items():
            e, i = _split(subset_coco.dataset["annotations"])
            print(f"  GT {grp:<13}: {e} eligible + {i} ignored")
        print(f"  Predictions    : {len(self._predictions)} entries accumulated")
        if self._predictions:
            p = self._predictions[0]
            print(f"  Sample pred[0] : image_id={p['image_id']} "
                  f"category_id={p['category_id']} "
                  f"score={p['score']:.4f} "
                  f"mask.size={p['segmentation']['size']}")
        print("[DEBUG/mask_ap] ──────────────────────────────────")

    def _eval(self, gt_coco):
        if not self._predictions:
            return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
        # COCOeval mutates predictions; deep-copy via JSON so subset evals
        # stay independent.
        preds = json.loads(json.dumps(self._predictions))
        with contextlib.redirect_stdout(io.StringIO()):
            pred_coco = gt_coco.loadRes(preds)
            ev = COCOeval(gt_coco, pred_coco, "segm")
            ev.params.catIds = self.eval_class_ids
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        return {
            "AP":   float(ev.stats[0]),
            "AP50": float(ev.stats[1]),
            "AP75": float(ev.stats[2]),
        }


def write_mask_ap_log(state, epoch):
    """Append one AP row per group to the configured metrics directory."""
    result = state.get("mask_ap_result")
    if result is None:
        return
    log_dir = state.get("log_dir")
    if log_dir is None:
        return

    os.makedirs(log_dir, exist_ok=True)
    filename = state.get("mask_ap_filename", "mask_ap.csv")
    csv_path = os.path.join(log_dir, filename)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "group", "AP", "AP50", "AP75"])
        if write_header:
            w.writeheader()
        for grp in ("overall", "occluded", "not_occluded"):
            d = result.get(grp)
            if d is None:
                continue
            w.writerow({
                "epoch": epoch, "group": grp,
                "AP":    d["AP"],
                "AP50":  d["AP50"],
                "AP75":  d["AP75"],
            })
