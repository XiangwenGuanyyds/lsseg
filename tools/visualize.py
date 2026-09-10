"""List test image IDs and visualise predictions from saved experiments."""

import argparse
import json
import math
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders import build_dataset
from models.build import build_model
from utils.output_paths import experiment_paths
from utils.saved_config import load_saved_config


def local_path(value, old_root, root):
    path = Path(value)
    if path.is_absolute():
        try:
            return root / path.relative_to(old_root)
        except ValueError:
            return path
    return root / path


def load_experiment(root, name, device, quiet=False):
    if name in (".", ".."):
        raise ValueError(f"Invalid experiment name: {name}")
    paths = experiment_paths(root, name)
    config_path = paths.test_config if paths.test_config.is_file() else paths.training_config
    if not config_path.is_file():
        raise FileNotFoundError(f"Saved config not found for {name}")
    saved = json.loads(config_path.read_text())
    cfg, _ = load_saved_config(config_path, root)
    old_root = Path(saved["values"].get("BASE_DIR", root))
    cfg.DEVICE = device
    cfg.DEBUG = {}
    cfg.DIAGNOSTICS = {}

    weights = paths.checkpoint
    metadata_path = config_path.with_name("metadata.json")
    if config_path == paths.test_config and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        recorded = (metadata.get("checkpoint") or {}).get("path")
        if recorded:
            weights = local_path(recorded, old_root, root)
    if not quiet:
        print(f"{name}: config {config_path}", flush=True)
    return cfg, weights


def list_experiments(root):
    output_dir = root / "outputs"
    available = []
    skipped = 0
    if output_dir.is_dir():
        for folder in sorted(output_dir.iterdir()):
            if not folder.is_dir():
                continue
            if not any((folder / mode / "config.json").is_file() for mode in ("training", "test")):
                continue
            try:
                _, weights = load_experiment(root, folder.name, "cpu", quiet=True)
                if not weights.is_file():
                    skipped += 1
                    continue
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                skipped += 1
                continue
            available.append(folder.name)
    print(f"Available experiments (saved config and checkpoint): {len(available)}")
    for name in available:
        print(name)
    if skipped:
        print(f"Skipped {skipped} experiments with missing checkpoints or invalid saved settings.")


def annotation_index(dataset):
    records = {}
    for info in dataset.data_infos:
        for ann in info["annotations"]:
            if int(ann["category_id"]) > 0:
                records[int(ann["id"])] = int(info["id"])
    return records


def selected_images(datasets, names, image_ids, ann_ids):
    ids = list(dict.fromkeys(image_ids or []))
    reference = annotation_index(datasets[0])
    for name, dataset in zip(names, datasets):
        records = annotation_index(dataset)
        missing = [ann_id for ann_id in ann_ids if ann_id not in records]
        if missing:
            raise ValueError(f"{name}: annotation IDs not in test set: {missing}. Use --list-ann-ids.")
        for ann_id in ann_ids:
            if records[ann_id] != reference[ann_id]:
                raise ValueError(f"Annotation ID {ann_id} belongs to different images across experiments")
    for ann_id in ann_ids:
        image_id = reference[ann_id]
        if image_ids and image_id not in ids:
            raise ValueError(f"Annotation ID {ann_id} belongs to image {image_id}, outside --image-ids")
        if image_id not in ids:
            ids.append(image_id)
    return ids


def list_annotations(dataset, name, ids=None):
    known = image_indices(dataset)
    if ids:
        missing = [image_id for image_id in ids if image_id not in known]
        if missing:
            raise ValueError(f"{name}: image IDs not in test set: {missing}")
    print(f"\n{name}: test annotation IDs")
    for info in sorted(dataset.data_infos, key=lambda item: int(item["id"])):
        if ids and int(info["id"]) not in ids:
            continue
        ann_ids = [str(ann["id"]) for ann in info["annotations"] if int(ann["category_id"]) > 0]
        print(f"Image {info['id']} ({Path(info['file_path']).name})")
        print(f"  Annotation IDs: {', '.join(ann_ids) or 'none'}")


def image_indices(dataset):
    return {int(info["id"]): index for index, info in enumerate(dataset.data_infos)}


def check_images(datasets, names, ids):
    maps = [image_indices(dataset) for dataset in datasets]
    for name, indices in zip(names, maps):
        missing = [image_id for image_id in ids if image_id not in indices]
        if missing:
            raise ValueError(f"{name}: image IDs not in test set: {missing}. Use --list-ids.")
    for image_id in ids:
        files = [Path(dataset.data_infos[indices[image_id]]["file_path"]).resolve()
                 for dataset, indices in zip(datasets, maps)]
        if len(set(files)) != 1:
            raise ValueError(f"Image ID {image_id} refers to different files across experiments")
        if not files[0].is_file():
            raise FileNotFoundError(f"Image not found: {files[0]}")
    return maps


def load_models(configs, weights, names):
    models = []
    for cfg, checkpoint, name in zip(configs, weights, names):
        print(f"{name}: loading {checkpoint}", flush=True)
        model = build_model(cfg)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        del state
        model.eval()
        models.append(model)
    return models


def predict(model, image, device, score_threshold):
    with torch.inference_mode():
        result = model([image.to(device)])[0]
    keep = (result["labels"] > 0) & (result["scores"] >= score_threshold)
    return {
        "boxes": result["boxes"][keep].cpu().numpy(),
        "scores": result["scores"][keep].cpu().numpy(),
        "masks": (result["masks"][keep, 0] >= 0.5).cpu().numpy(),
    }


def draw_masks(ax, image, masks, colour, bounds):
    x1, y1, x2, y2 = bounds
    ax.imshow(image[y1:y2, x1:x2])
    cropped = masks[:, y1:y2, x1:x2]
    overlay = np.zeros((y2 - y1, x2 - x1, 4))
    if len(cropped):
        overlay[cropped.any(axis=0)] = [*colour, 0.4]
    ax.imshow(overlay)
    for mask in cropped:
        if mask.any():
            # Padding also gives a contour for a mask that fills the crop.
            padded = np.pad(mask, 1)
            ax.contour(np.arange(-1, mask.shape[1] + 1), np.arange(-1, mask.shape[0] + 1),
                       padded, levels=[0.5], colors=[colour], linewidths=0.7)
    ax.set_xlim(-0.5, x2 - x1 - 0.5)
    ax.set_ylim(y2 - y1 - 0.5, -0.5)
    ax.axis("off")


def render(image, target, predictions, names, image_id, bounds=None, ann_id=None):
    height, width = image.shape[:2]
    bounds = bounds or (0, 0, width, height)
    count = len(names) + 1
    columns = min(count, 3)
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.8 * rows), squeeze=False)
    gt_masks = target["masks"][target["labels"] > 0].cpu().numpy().astype(bool)
    draw_masks(axes.flat[0], image, gt_masks, (0.15, 0.85, 0.3), bounds)
    axes.flat[0].set_title("Ground truth", fontsize=10)
    x1, y1, x2, y2 = bounds
    for ax, name, prediction in zip(list(axes.flat)[1:], names, predictions):
        draw_masks(ax, image, prediction["masks"], (0.1, 0.5, 1.0), bounds)
        for box, score in zip(prediction["boxes"], prediction["scores"]):
            bx1, by1, bx2, by2 = box
            if bx2 <= x1 or by2 <= y1 or bx1 >= x2 or by1 >= y2:
                continue
            ax.add_patch(Rectangle((bx1 - x1, by1 - y1), bx2 - bx1, by2 - by1,
                                   fill=False, edgecolor="white", linewidth=0.6))
            ax.text(max(bx1 - x1, 0), max(by1 - y1, 0), f"{score:.2f}",
                    color="white", fontsize=7, va="top", clip_on=True,
                    bbox={"facecolor": "black", "alpha": 0.5, "pad": 1, "edgecolor": "none"})
        ax.set_title(textwrap.fill(name, 42), fontsize=10)
    for ax in list(axes.flat)[count:]:
        ax.set_visible(False)
    title = f"Image ID: {image_id}"
    if ann_id is not None:
        title += f" | Annotation ID: {ann_id}"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def crop_bounds(box, height, width, padding):
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    return (max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy)),
            min(width, math.ceil(x2 + dx)), min(height, math.ceil(y2 + dy)))


def save_figure(fig, root, names, image_id, ann_id=None):
    suffix = f"image_{image_id}"
    if ann_id is not None:
        suffix += f"_ann_{ann_id}"
    if len(names) == 1:
        folder = experiment_paths(root, names[0]).analysis / "visualizations"
    else:
        folder = root / "local_record/visualizations"
        suffix = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{suffix}"
    path = folder / f"{suffix}.png"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
    print(f"Saved {path.resolve()}", flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="*", help="Experiment names under outputs/")
    parser.add_argument("--image-ids", "--image-id", type=int, nargs="+", help="Test image IDs")
    parser.add_argument("--ann-ids", type=int, nargs="+", help="Generate crops for these annotation IDs")
    listing = parser.add_mutually_exclusive_group()
    listing.add_argument("--list-ids", action="store_true", help="List test image IDs and filenames")
    listing.add_argument("--list-ann-ids", action="store_true", help="List annotation IDs, optionally filtered by --image-ids")
    listing.add_argument("--list-experiments", action="store_true", help="List experiments with saved configs and checkpoints")
    parser.add_argument("--view", choices=("full", "crops", "both"),
                        help="Default: crops with --ann-ids, otherwise full")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--padding", type=float, default=0.3, help="Crop padding relative to GT box size")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Default: CUDA when available, otherwise CPU")
    args = parser.parse_args()
    if args.list_experiments:
        list_experiments(PROJECT_ROOT)
        return
    if not args.experiments:
        parser.error("Provide an experiment name, or use --list-experiments")
    if not (args.list_ids or args.list_ann_ids or args.image_ids or args.ann_ids):
        parser.error("Provide --image-ids or --ann-ids, or use a list option")
    if args.ann_ids and (args.list_ids or args.list_ann_ids):
        parser.error("Use --ann-ids for generating crops, separately from list options")
    if args.ann_ids and args.view == "full":
        parser.error("--ann-ids is used with --view crops or both")
    if not 0 <= args.score_threshold <= 1 or not math.isfinite(args.padding) or args.padding < 0:
        parser.error("Score threshold must be between 0 and 1; padding must be non-negative")
    names = list(dict.fromkeys(args.experiments))
    view = args.view or ("crops" if args.ann_ids else "full")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        settings = [load_experiment(PROJECT_ROOT, name, device) for name in names]
        configs = [item[0] for item in settings]
        weights = [item[1] for item in settings]
        datasets = [build_dataset(cfg, mode="test") for cfg in configs]
        if args.list_ids:
            for name, dataset in zip(names, datasets):
                print(f"\n{name}: {len(dataset.data_infos)} test images", flush=True)
                for info in sorted(dataset.data_infos, key=lambda info: int(info["id"])):
                    print(f"{info['id']}  {Path(info['file_path']).name}")
            return
        if args.list_ann_ids:
            for name, dataset in zip(names, datasets):
                list_annotations(dataset, name, args.image_ids)
            return

        if args.ann_ids:
            ids = selected_images(datasets, names, args.image_ids, args.ann_ids)
        else:
            ids = list(dict.fromkeys(args.image_ids))
        indices = check_images(datasets, names, ids)
        if any(cfg.TEST_PIPELINE != configs[0].TEST_PIPELINE for cfg in configs[1:]):
            raise ValueError("Experiments use different test image transforms; compare them separately")
        for checkpoint in weights:
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        models = load_models(configs, weights, names)
        for image_id in ids:
            samples = [dataset[index[image_id]] for dataset, index in zip(datasets, indices)]
            image = (samples[0][0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            target = samples[0][1]
            predictions = []
            for name, model, (tensor, _) in zip(names, models, samples):
                print(f"{name}: predicting image {image_id}", flush=True)
                predictions.append(predict(model, tensor, device, args.score_threshold))
            if view in ("full", "both"):
                fig = render(image, target, predictions, names, image_id)
                save_figure(fig, PROJECT_ROOT, names, image_id)
            if view in ("crops", "both"):
                for box, ann_id, label in zip(target["boxes"], target["ann_ids"], target["labels"]):
                    if label <= 0:
                        continue
                    if args.ann_ids and int(ann_id) not in args.ann_ids:
                        continue
                    bounds = crop_bounds(box.tolist(), *image.shape[:2], args.padding)
                    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                        continue
                    fig = render(image, target, predictions, names, image_id, bounds, int(ann_id))
                    save_figure(fig, PROJECT_ROOT, names, image_id, int(ann_id))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
