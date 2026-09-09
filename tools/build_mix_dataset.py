"""Build the per_scene/mix dataset by per-prefix subsampling within each raw split.

USAGE
    python tools/build_mix_dataset.py
    python tools/build_mix_dataset.py --seed 1
    python tools/build_mix_dataset.py --train 500 --val 100 --test 100 --seed 0

DESIGN
    For each (raw_split, out_split) pair — (train, train), (valid, val),
    (test, test) — produce one COCO file:

        mix/<out_split>.coco.json   ← drawn entirely from raw/<raw_split>/

    Within that pair, every prefix (pig / frame / stream / I) contributes
    target//n images, picked by the same stride-along-embedded-frame-index
    used by the baselines (decorrelates video frames). If a prefix has
    fewer than its share, it gives all it has and a global random sample
    over the same raw_split's leftover pool fills the gap.

    This means:
      - mix/val.coco.json's images all live in dataset/raw/pigs/pig/valid/
      - mix/test.coco.json's images all live in dataset/raw/pigs/pig/test/
      - mix/train.coco.json's images all live in dataset/raw/pigs/pig/train/
    so SPLIT_TO_RAW_SUBDIR in tools/annotate_occlusion_round1.py covers it directly,
    and at open-source release the images for split S can be cp-ed from
    raw/<S-or-valid>/<file_name>.

    Raw image_id and ann_id are kept verbatim (same as the baseline
    builder). Cross-split ann_id collisions are inherited from raw at the
    natural rate (~9% for pig val/test) — not amplified.

OUTPUT
    dataset/per_scene/mix/train.coco.json
    dataset/per_scene/mix/val.coco.json
    dataset/per_scene/mix/test.coco.json
    dataset/per_scene/mix/build_manifest.json   (audit trail)

NOTES
    - No image copying.
    - All categories (id 0/1/2) preserved verbatim. CATEGORY_REMAP={2:1}
      in the cfg still drops cat 0/1 at training time.
    - Zero-annotation source images are excluded before stride, matching
      the baseline builder's policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT  = REPO_ROOT / "dataset" / "raw" / "pigs" / "pig"
OUT_DIR   = REPO_ROOT / "dataset" / "per_scene" / "mix"

SPLITS = [("train", "train"), ("valid", "val"), ("test", "test")]


# How to recognize and sort each prefix. "I" added to the baseline set.
SCENES = {
    "pig":    {"regex": re.compile(r"^pig-(\d+)-_jpg"),   "sort": "int"},
    "frame":  {"regex": re.compile(r"^frame_(\d+)_jpg"),  "sort": "int"},
    "stream": {"regex": re.compile(r"^stream_"),          "sort": "name"},
    "I":      {"regex": re.compile(r"^I(\d+)_jpg"),       "sort": "int"},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=int, default=500)
    p.add_argument("--val",   type=int, default=100)
    p.add_argument("--test",  type=int, default=100)
    p.add_argument("--seed",  type=int, default=42)
    return p.parse_args()


# ----------------------------------------------------------------------
# Sampling helpers
# ----------------------------------------------------------------------
def filter_and_sort(images, scene_def, annotated_ids):
    """Pick images that match the scene AND have at least one source ann."""
    pat  = scene_def["regex"]
    mode = scene_def["sort"]
    if mode == "int":
        keyed = []
        for im in images:
            if im["id"] not in annotated_ids:
                continue
            m = pat.match(im["file_name"])
            if m:
                keyed.append((int(m.group(1)), im["file_name"], im))
        keyed.sort(key=lambda x: (x[0], x[1]))
        return [k[2] for k in keyed]
    return sorted(
        [im for im in images
         if im["id"] in annotated_ids and pat.match(im["file_name"])],
        key=lambda im: im["file_name"],
    )


def stride_indices(n_have, n_want):
    """K evenly-spaced unique indices in [0, n_have-1]."""
    if n_want >= n_have:
        return list(range(n_have))
    return np.linspace(0, n_have - 1, n_want).round().astype(int).tolist()


def assigned_prefix(file_name):
    """Return which scene prefix this file_name belongs to, or None."""
    for name, sd in SCENES.items():
        if sd["regex"].match(file_name):
            return name
    return None


# ----------------------------------------------------------------------
# core
# ----------------------------------------------------------------------
def build_one_split(raw_split, out_split, target, seed):
    """Build mix/<out_split>.coco.json by drawing from raw/<raw_split>."""
    raw_ann = RAW_ROOT / raw_split / "_annotations.coco.json"
    with raw_ann.open() as f:
        coco = json.load(f)

    annotated_ids = {a["image_id"] for a in coco["annotations"]}

    # Per-prefix sorted+annotated pool
    pools = {}
    for name, sd in SCENES.items():
        pools[name] = filter_and_sort(coco["images"], sd, annotated_ids)

    n = len(SCENES)
    per_prefix = target // n

    # Stride-pick from each prefix; collect the rest as the leftover pool
    # for global fill.
    selected_ids = set()
    selected_imgs = []
    leftover_imgs = []
    per_prefix_report = {}

    for name, pool in pools.items():
        idxs = stride_indices(len(pool), per_prefix)
        kept = [pool[i] for i in idxs]
        kept_id_set = {im["id"] for im in kept}
        selected_ids.update(kept_id_set)
        selected_imgs.extend(kept)
        for im in pool:
            if im["id"] not in kept_id_set:
                leftover_imgs.append(im)
        per_prefix_report[name] = {
            "pool_size": len(pool),
            "kept":      len(kept),
            "stride":    (len(pool) / per_prefix) if per_prefix else None,
        }

    # Global random fill from leftovers (same raw_split only).
    shortfall = target - len(selected_imgs)
    if shortfall > 0:
        rng = random.Random(seed + hash(out_split) % 10_000)
        rng.shuffle(leftover_imgs)
        avail = min(shortfall, len(leftover_imgs))
        fill = leftover_imgs[:avail]
        selected_imgs.extend(fill)
        selected_ids.update(im["id"] for im in fill)
        if avail < shortfall:
            print(f"[mix {out_split}] WARNING: short by {shortfall - avail} "
                  f"after leftover exhausted")
        fill_count = avail
    else:
        fill_count = 0

    # Slice annotations for selected images. ann ids and image ids are
    # kept verbatim from raw.
    kept_anns = [a for a in coco["annotations"] if a["image_id"] in selected_ids]

    out_coco = {
        "info":        coco.get("info", {}),
        "licenses":    coco.get("licenses", []),
        "categories":  coco.get("categories", []),
        "images":      selected_imgs,
        "annotations": kept_anns,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{out_split}.coco.json"
    with out_path.open("w") as f:
        json.dump(out_coco, f)

    h = hashlib.sha1(",".join(sorted(im["file_name"] for im in selected_imgs))
                     .encode()).hexdigest()[:12]
    return {
        "raw_split":      raw_split,
        "out_split":      out_split,
        "target":         target,
        "kept":           len(selected_imgs),
        "n_anns":         len(kept_anns),
        "per_prefix":     per_prefix_report,
        "global_fill":    fill_count,
        "kept_files_sha": h,
        "out_file":       str(out_path.relative_to(REPO_ROOT)),
    }


def main():
    args = parse_args()
    targets = {"train": args.train, "val": args.val, "test": args.test}

    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "policy":   "per-prefix stride-subsample within each raw_split "
                    "(target//n per prefix); shortfall filled by random "
                    "sampling from the same raw_split's leftover pool. "
                    "raw image_id / ann_id preserved. raw subdir mapping "
                    "follows the baseline convention (val→raw/valid, "
                    "test→raw/test, train→raw/train).",
        "seed":     args.seed,
        "targets":  targets,
        "splits":   [],
    }

    for raw_split, out_split in SPLITS:
        r = build_one_split(raw_split, out_split, targets[out_split], args.seed)
        manifest["splits"].append(r)
        per_pref = ", ".join(
            f"{k}={v['kept']}/{v['pool_size']}" for k, v in r["per_prefix"].items()
        )
        print(f"[mix {out_split:5s}] target={r['target']:4d}  "
              f"kept={r['kept']:4d}  anns={r['n_anns']:5d}  "
              f"fill={r['global_fill']}  | {per_pref}")

    mpath = OUT_DIR / "build_manifest.json"
    with mpath.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nmanifest -> {mpath.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
