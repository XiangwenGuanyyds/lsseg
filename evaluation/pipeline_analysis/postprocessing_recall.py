"""Compatibility imports for the renamed recall-retention module."""

from .recall_retention import (
    accumulate_postprocessing_recall,
    write_postprocessing_recall,
)

__all__ = [
    "accumulate_postprocessing_recall",
    "write_postprocessing_recall",
]
