"""Stable command-line interface for training, testing, and result analysis."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.output_paths import experiment_paths


MAIN_PATH = PROJECT_ROOT / "main.py"


def _run_main(arguments):
    command = [sys.executable, str(MAIN_PATH), *arguments]
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _training_metadata(experiment_name):
    paths = experiment_paths(PROJECT_ROOT, experiment_name)
    metadata = _load_json(paths.training_metadata) or {}
    config = _load_json(paths.training_config) or {}
    config_name = (
        metadata.get("config", {}).get("registry_name")
        or config.get("registry_name")
    )
    seed = config.get("values", {}).get("SEED")
    return paths, config_name, seed


def _append_override(command, option, value):
    if value is not None:
        command.extend([option, str(value)])


def train(args):
    command = [
        "--mode", "train",
        "--exp_name", args.experiment,
        "--seed", str(args.seed),
    ]
    if args.config_file:
        command.extend(["--config-file", str(Path(args.config_file).resolve())])
    else:
        command.extend(["--config", args.config])
    _append_override(command, "--num_workers", args.num_workers)
    _append_override(command, "--batch_size", args.batch_size)
    _run_main(command)


def test(args):
    paths, _, saved_seed = _training_metadata(args.experiment)
    if args.config_file:
        config_args = ["--config-file", str(Path(args.config_file).resolve())]
    elif args.config:
        config_args = ["--config", args.config]
    elif paths.training_config.is_file():
        config_args = ["--config-file", str(paths.training_config)]
    else:
        raise SystemExit(
            "No saved training config exists. Pass --config or --config-file explicitly."
        )

    weights = Path(args.weights) if args.weights else paths.checkpoint
    if not weights.is_file():
        raise SystemExit(f"checkpoint not found: {weights}")

    result_name = args.result_name or args.experiment
    command = [
        "--mode", "test",
        *config_args,
        "--weights", str(weights),
        "--exp_name", result_name,
    ]
    seed = args.seed
    if seed is None and not args.config_file:
        seed = saved_seed
    _append_override(command, "--seed", seed)
    _append_override(command, "--num_workers", args.num_workers)
    _append_override(command, "--batch_size", args.batch_size)
    if args.no_diagnostics:
        command.append("--no-diagnostics")
    _run_main(command)


def _report_rows(experiment_name, refresh=False):
    paths = experiment_paths(PROJECT_ROOT, experiment_name)
    report_path = paths.test_metrics / "coco_ap_report.csv"
    if refresh or not report_path.exists():
        sys.path.insert(0, str(PROJECT_ROOT))
        from evaluation.coco_ap_report import write_coco_ap_report
        try:
            report_path, _ = write_coco_ap_report(paths.test, paths.test_metrics)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
    with report_path.open(newline="") as f:
        return list(csv.DictReader(f))


def ap(args):
    rows = _report_rows(args.experiment, refresh=args.refresh)
    print(f"COCO AP: {args.experiment}")
    print(f"{'group':16s}{'type':>8s}{'AP':>10s}{'AP50':>10s}{'AP75':>10s}")
    for row in rows:
        print(
            f"{row['group']:16s}{row['iou_type']:>8s}"
            f"{float(row['AP']):>10.4f}"
            f"{float(row['AP50']):>10.4f}"
            f"{float(row['AP75']):>10.4f}"
        )


def pixel_metrics(args):
    if not 0.0 <= args.score_threshold <= 1.0:
        raise SystemExit("--score-threshold must be between 0 and 1")
    paths = experiment_paths(PROJECT_ROOT, args.experiment)
    try:
        from evaluation.foreground_metrics import (
            evaluate_foreground_metrics,
            write_overall_pixel_metrics,
        )

        metrics = evaluate_foreground_metrics(
            paths.test,
            score_threshold=args.score_threshold,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    output_path = write_overall_pixel_metrics(
        paths.pixel_metrics,
        args.experiment,
        metrics,
        score_threshold=args.score_threshold,
        ground_truth_path=paths.ground_truth / "overall.json",
    )
    print(f"Overall pixel metrics: {args.experiment}")
    print(f"  Global foreground IoU : {metrics['global_foreground_iou']:.4f}")
    print(f"  Pixel Precision       : {metrics['pixel_precision']:.4f}")
    print(f"  Pixel Recall          : {metrics['pixel_recall']:.4f}")
    print(f"  Pixel F1              : {metrics['pixel_f1']:.4f}")
    print(f"wrote {output_path}")


def analyze_pipeline(args):
    paths = experiment_paths(PROJECT_ROOT, args.experiment)
    try:
        from evaluation.pipeline_analysis.report import (
            load_pipeline_metrics,
            print_summary,
        )

        result = load_pipeline_metrics(paths.pipeline_analysis, args.experiment)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    print_summary(args.experiment, result)


def analyze_mask_errors(args):
    paths = experiment_paths(PROJECT_ROOT, args.experiment)
    try:
        from evaluation.mask_error_analysis.evaluator import (
            evaluate_mask_errors,
            print_summary,
        )

        result = evaluate_mask_errors(
            paths.test,
            args.experiment,
            output_dir=paths.mask_errors,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    print_summary(args.experiment, result)


def info(args):
    paths, config_name, seed = _training_metadata(args.experiment)
    test_metadata = _load_json(paths.test_metadata) or {}
    test_config = _load_json(paths.test_config) or {}
    if config_name is None:
        config_name = test_metadata.get("config", {}).get("registry_name")
    if seed is None:
        seed = test_config.get("values", {}).get("SEED")

    print(f"experiment : {args.experiment}")
    print(f"config     : {config_name or 'unknown'}")
    print(f"seed       : {seed if seed is not None else 'unknown'}")
    print(f"training   : {paths.training} ({'exists' if paths.training.is_dir() else 'missing'})")
    print(f"checkpoint : {paths.checkpoint} ({'exists' if paths.checkpoint.is_file() else 'missing'})")
    print(f"test       : {paths.test} ({'exists' if paths.test.is_dir() else 'missing'})")
    print(f"analysis   : {paths.analysis} ({'exists' if paths.analysis.is_dir() else 'missing'})")
    report_path = paths.test_metrics / "coco_ap_report.csv"
    print(
        f"COCO report: {report_path} "
        f"({'exists' if report_path.is_file() else 'missing'})"
    )
    pixel_path = paths.pixel_metrics / "overall.csv"
    print(
        f"pixel metrics: {pixel_path} "
        f"({'exists' if pixel_path.is_file() else 'missing'})"
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Unified experiment workflow for LSSeg.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train one experiment")
    train_parser.add_argument("experiment", help="experiment name under outputs/")
    train_source = train_parser.add_mutually_exclusive_group(required=True)
    train_source.add_argument("--config", help="registered public config name")
    train_source.add_argument("--config-file", help="saved experiment config.json")
    train_parser.add_argument("--seed", required=True, type=int)
    train_parser.add_argument("--num-workers", type=int)
    train_parser.add_argument("--batch-size", type=int)
    train_parser.set_defaults(func=train)

    test_parser = subparsers.add_parser("test", help="test a trained experiment")
    test_parser.add_argument("experiment", help="source training experiment under outputs/")
    test_source = test_parser.add_mutually_exclusive_group()
    test_source.add_argument("--config", help="use a public test configuration")
    test_source.add_argument("--config-file", help="use another saved config.json")
    test_parser.add_argument("--weights", help="override checkpoint path")
    test_parser.add_argument("--result-name", help="name of the test output experiment")
    test_parser.add_argument("--seed", type=int, help="override saved training seed")
    test_parser.add_argument("--num-workers", type=int)
    test_parser.add_argument("--batch-size", type=int)
    test_parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="run standard test metrics without pipeline diagnostics",
    )
    test_parser.set_defaults(func=test)

    ap_parser = subparsers.add_parser("ap", help="print bbox and segm COCO AP")
    ap_parser.add_argument("experiment", help="experiment name under outputs/")
    ap_parser.add_argument(
        "--refresh", action="store_true", help="recompute the report from saved predictions",
    )
    ap_parser.set_defaults(func=ap)

    pixel_parser = subparsers.add_parser(
        "pixel-metrics",
        help="compute overall foreground pixel metrics from saved test results",
    )
    pixel_parser.add_argument("experiment", help="experiment name under outputs/")
    pixel_parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="additional score filter applied to saved predictions (default: 0.0)",
    )
    pixel_parser.set_defaults(func=pixel_metrics)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="run one analysis group for a completed test result",
    )
    analysis_parsers = analyze_parser.add_subparsers(
        dest="analysis",
        required=True,
    )

    pipeline_parser = analysis_parsers.add_parser(
        "pipeline",
        help="show RPN and post-processing analysis from test outputs",
    )
    pipeline_parser.add_argument(
        "experiment",
        help="experiment name under outputs/",
    )
    pipeline_parser.set_defaults(func=analyze_pipeline)

    mask_error_parser = analysis_parsers.add_parser(
        "mask-errors",
        help="compute boundary-distance and instance-leakage analysis",
    )
    mask_error_parser.add_argument(
        "experiment",
        help="experiment name under outputs/",
    )
    mask_error_parser.set_defaults(func=analyze_mask_errors)

    info_parser = subparsers.add_parser("info", help="show experiment paths and metadata")
    info_parser.add_argument("experiment")
    info_parser.set_defaults(func=info)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
