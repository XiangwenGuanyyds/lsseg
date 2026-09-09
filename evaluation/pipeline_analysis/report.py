"""Read and check RPN coverage and post-processing recall CSV files."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from evaluation.artifact_checks import require_test_artifacts


REQUIRED_ARTIFACTS = {
    "rpn_coverage": ("rpn_coverage.csv",),
    "postprocessing_recall": (
        "postprocessing_recall.csv",
        "roi_coverage_loss.csv",
    ),
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _close(left: float, right: float, tolerance: float = 1e-8) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _validate_rpn_coverage(rows: list[dict[str, str]]) -> None:
    required = {"threshold", "group", "n_gt", "n_covered", "coverage_rate"}
    for index, row in enumerate(rows, start=2):
        if not required.issubset(row):
            raise ValueError(f"Invalid RPN coverage schema at CSV row {index}")
        n_gt = int(row["n_gt"])
        n_covered = int(row["n_covered"])
        coverage = float(row["coverage_rate"])
        expected = n_covered / n_gt if n_gt else 0.0
        if n_covered > n_gt or not _close(coverage, expected):
            raise ValueError(f"Invalid RPN coverage values at CSV row {index}")


def _validate_postprocessing_recall(rows: list[dict[str, str]]) -> None:
    required = {
        "iou_threshold",
        "group",
        "num_gt",
        "num_pre_matched_gt",
        "num_post_matched_gt",
        "recall_pre",
        "recall_post",
        "recall_retention",
    }
    for index, row in enumerate(rows, start=2):
        if not required.issubset(row):
            raise ValueError(
                f"Invalid post-processing recall schema at CSV row {index}"
            )
        n_gt = int(row["num_gt"])
        n_pre = int(row["num_pre_matched_gt"])
        n_post = int(row["num_post_matched_gt"])
        recall_pre = float(row["recall_pre"])
        recall_post = float(row["recall_post"])
        retention = float(row["recall_retention"])
        expected_pre = n_pre / n_gt if n_gt else math.nan
        expected_post = n_post / n_gt if n_gt else math.nan
        expected_retention = (
            expected_post / expected_pre if expected_pre > 0 else math.nan
        )
        if n_pre > n_gt or n_post > n_gt:
            raise ValueError(
                f"Invalid post-processing recall counts at CSV row {index}"
            )
        for actual, expected in (
            (recall_pre, expected_pre),
            (recall_post, expected_post),
            (retention, expected_retention),
        ):
            if not (
                (math.isnan(actual) and math.isnan(expected))
                or _close(actual, expected)
            ):
                raise ValueError(
                    f"Invalid post-processing recall values at CSV row {index}"
                )


def load_pipeline_metrics(result_dir: Path, experiment: str) -> dict:
    """Load and check both pipeline metric tables for one experiment."""
    artifacts = require_test_artifacts(
        result_dir,
        experiment,
        REQUIRED_ARTIFACTS,
    )
    rpn_rows = _read_rows(artifacts["rpn_coverage"])
    postprocessing_rows = _read_rows(artifacts["postprocessing_recall"])
    _validate_rpn_coverage(rpn_rows)
    _validate_postprocessing_recall(postprocessing_rows)
    return {
        "artifacts": artifacts,
        "rpn_coverage": rpn_rows,
        "postprocessing_recall": postprocessing_rows,
    }


# Compatibility alias for existing external callers.
evaluate_result_directory = load_pipeline_metrics


def print_summary(experiment: str, result: dict) -> None:
    print(f"Pipeline analysis: {experiment}")
    print("\nRPN coverage")
    print(f"{'IoU':>6s}  {'group':16s}  {'coverage':>10s}")
    for row in result["rpn_coverage"]:
        group = "overall" if row["group"] == "all" else row["group"]
        print(
            f"{float(row['threshold']):>6.1f}  {group:16s}  "
            f"{float(row['coverage_rate']):>10.4f}"
        )

    print("\nPost-processing recall")
    print(
        f"{'IoU':>6s}  {'group':16s}  {'before':>10s}  "
        f"{'after':>10s}  {'retention':>10s}"
    )
    for row in result["postprocessing_recall"]:
        print(
            f"{float(row['iou_threshold']):>6.1f}  {row['group']:16s}  "
            f"{float(row['recall_pre']):>10.4f}  "
            f"{float(row['recall_post']):>10.4f}  "
            f"{float(row['recall_retention']):>10.4f}"
        )

    for name, path in result["artifacts"].items():
        print(f"{name}: {path}")
