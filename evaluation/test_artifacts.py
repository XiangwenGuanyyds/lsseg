"""Compatibility imports for the renamed artifact checks module."""

from .artifact_checks import MissingTestArtifactsError, require_test_artifacts

__all__ = ["MissingTestArtifactsError", "require_test_artifacts"]
