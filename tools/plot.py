"""Plot training loss or validation AP for one or more experiments."""

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.output_paths import experiment_paths

GROUPS = ("overall", "occluded", "not_occluded")


def read_log(path, kind):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    required = {"epoch", "group"} if kind == "val" else {"epoch"}
    if not required.issubset(fields):
        raise ValueError(f"Missing columns in {path}: {sorted(required - set(fields))}")
    if not rows:
        raise ValueError(f"Empty log: {path}")

    available = [f for f in fields if f not in ("epoch", "group")]
    records = {}
    for line, row in enumerate(rows, 2):
        try:
            epoch = int(row["epoch"])
            group = row["group"] if kind == "val" else ""
            values = {field: float(row[field]) for field in available}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError("non-finite metric")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid data in {path}, line {line}") from exc
        # Resumed logs can repeat an epoch. Keep the last recorded row.
        records[epoch, group] = values
    return available, records


def load_experiments(root, names, kind):
    filename = "train_losses.csv" if kind == "loss" else "validation_mask_ap.csv"
    logs = {}
    for name in dict.fromkeys(names):
        if name in (".", ".."):
            raise ValueError(f"Invalid experiment name: {name}")
        path = experiment_paths(root, name).training_metrics / filename
        logs[name] = read_log(path, kind)
    return logs


def make_figure(logs, kind, metrics=None, groups=None):
    available = list(dict.fromkeys(field for fields, _ in logs.values() for field in fields))
    metrics = metrics or (["total"] if kind == "loss" else ["AP", "AP50", "AP75"])
    if metrics == ["all"]:
        metrics = available
    else:
        metrics = list(dict.fromkeys(metrics))
    if not metrics:
        raise ValueError("No metrics found in the logs")
    for metric in metrics:
        if metric not in available:
            raise ValueError(f"Unknown metric: {metric}. Available: {', '.join(available)}")

    if kind == "val":
        groups = groups or list(GROUPS)
    else:
        groups = [""]
    groups = list(dict.fromkeys(groups))
    for name, (fields, records) in logs.items():
        missing = [metric for metric in metrics if metric not in fields]
        if missing:
            print(f"{name}: no {', '.join(missing)} column; those curves are skipped")
        if not any(metric in fields for metric in metrics):
            raise ValueError(f"{name}: none of the requested metrics are available")
        for group in groups:
            if not any(key[1] == group for key in records):
                raise ValueError(f"{name}: no validation rows for {group}")

    panels = [(group, metric) for group in groups for metric in metrics]
    ncols = min(3, len(metrics))
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    styles = ("-", "--", "-.", ":")
    handles = {}
    for ax, (group, metric) in zip(axes.flat, panels):
        for i, (name, (fields, records)) in enumerate(logs.items()):
            if metric not in fields:
                continue
            points = sorted((epoch, values[metric]) for (epoch, g), values in records.items() if g == group)
            line, = ax.plot([p[0] for p in points], [p[1] for p in points],
                            color=f"C{i % 10}", linestyle=styles[(i // 10) % len(styles)],
                            linewidth=1.5, label=name)
            handles.setdefault(name, line)
        title = f"{group.replace('_', ' ').capitalize()} - {metric}" if group else metric
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mask AP" if kind == "val" else "Loss (epoch mean)")
        if kind == "val":
            ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.legend([handles[name] for name in logs], list(logs),
               loc="upper center", bbox_to_anchor=(0.5, 0), frameon=False, fontsize=9)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="+", help="Experiment names under outputs/")
    parser.add_argument("--type", choices=("loss", "val"), default="loss",
                        help="Plot training loss or validation mask AP (default: loss)")
    parser.add_argument("--metrics", nargs="+", help="CSV column names, or all")
    parser.add_argument("--groups", nargs="+", choices=GROUPS,
                        help="Validation groups (default: all three groups)")
    parser.add_argument("--output", type=Path, help="Output PNG path")
    args = parser.parse_args()
    if args.groups and args.type != "val":
        parser.error("--groups is used with --type val")
    if args.output and args.output.suffix.lower() != ".png":
        parser.error("--output must end in .png")

    fig = None
    try:
        logs = load_experiments(PROJECT_ROOT, args.experiments, args.type)
        fig = make_figure(logs, args.type, args.metrics, args.groups)
        filename = "training_loss.png" if args.type == "loss" else "validation_mask_ap.png"
        if len(logs) == 1:
            folder = experiment_paths(PROJECT_ROOT, next(iter(logs))).figures
        else:
            folder = PROJECT_ROOT / "local_record" / "plots"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}_{filename}"
        output = args.output or folder / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved {output.resolve()}")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    finally:
        if fig is not None:
            plt.close(fig)


if __name__ == "__main__":
    main()
