"""Compatibility imports for the renamed foreground metrics module."""

from .foreground_metrics import (
    OUTPUT_FIELDS,
    compute_overall_pixel_metrics,
    evaluate_foreground_metrics,
    evaluate_result_directory,
    load_ground_truth,
    load_predictions,
    write_overall_pixel_metrics,
)

__all__ = [
    "OUTPUT_FIELDS",
    "compute_overall_pixel_metrics",
    "evaluate_foreground_metrics",
    "evaluate_result_directory",
    "load_ground_truth",
    "load_predictions",
    "write_overall_pixel_metrics",
]
