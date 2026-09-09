"""Mask prediction error analysis from saved test outputs."""

from .evaluator import evaluate_mask_errors, print_summary

evaluate_result_directory = evaluate_mask_errors

__all__ = ["evaluate_mask_errors", "evaluate_result_directory", "print_summary"]
