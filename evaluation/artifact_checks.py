"""Check that the test files needed for analysis exist."""

from pathlib import Path
from typing import Mapping, Sequence


class MissingTestArtifactsError(FileNotFoundError):
    """Required test files are missing."""


def require_test_artifacts(
    result_dir: Path,
    experiment: str,
    requirements: Mapping[str, Sequence[str]],
) -> dict[str, Path]:
    """Return paths to required test files; raise an error if any are missing.

    Each requirement can list alternative filenames, checked in order.
    """
    result_dir = Path(result_dir)
    resolved: dict[str, Path] = {}
    missing: list[str] = []

    for key, filenames in requirements.items():
        candidates = [result_dir / filename for filename in filenames]
        existing = next((path for path in candidates if path.is_file()), None)
        if existing is None:
            missing.append(" or ".join(str(path) for path in candidates))
        else:
            resolved[key] = existing

    if missing:
        lines = [
            f"Required test outputs are missing for: {experiment}",
            "",
            "Missing:",
            *(f"  {item}" for item in missing),
            "",
            "Run the test first:",
            f"  ./lsseg test {experiment}",
        ]
        raise MissingTestArtifactsError("\n".join(lines))

    return resolved
