"""Canonical filesystem layout for experiment outputs."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentPaths:
    """Paths belonging to one named experiment under ``outputs/``."""

    project_root: Path
    experiment: str

    def __post_init__(self):
        object.__setattr__(self, "project_root", Path(self.project_root))
        if not self.experiment or Path(self.experiment).name != self.experiment:
            raise ValueError(f"invalid experiment name: {self.experiment!r}")

    @property
    def root(self) -> Path:
        return self.project_root / "outputs" / self.experiment

    @property
    def training(self) -> Path:
        return self.root / "training"

    @property
    def training_metrics(self) -> Path:
        return self.training / "metrics"

    @property
    def checkpoint(self) -> Path:
        return self.training / "checkpoint.pth"

    @property
    def training_config(self) -> Path:
        return self.training / "config.json"

    @property
    def training_metadata(self) -> Path:
        return self.training / "metadata.json"

    @property
    def test(self) -> Path:
        return self.root / "test"

    @property
    def test_metrics(self) -> Path:
        return self.test / "metrics"

    @property
    def ground_truth(self) -> Path:
        return self.test / "ground_truth"

    @property
    def test_config(self) -> Path:
        return self.test / "config.json"

    @property
    def test_metadata(self) -> Path:
        return self.test / "metadata.json"

    @property
    def analysis(self) -> Path:
        return self.root / "analysis"

    @property
    def pipeline_analysis(self) -> Path:
        return self.analysis / "pipeline"

    @property
    def mask_errors(self) -> Path:
        return self.analysis / "mask_errors"

    @property
    def pixel_metrics(self) -> Path:
        return self.analysis / "pixel_metrics"

    @property
    def figures(self) -> Path:
        return self.analysis / "figures"


def comparisons_dir(project_root) -> Path:
    """Directory for reports that combine multiple experiments."""
    return Path(project_root) / "outputs" / "comparisons"


def experiment_paths(project_root, experiment: str) -> ExperimentPaths:
    return ExperimentPaths(Path(project_root), experiment)
