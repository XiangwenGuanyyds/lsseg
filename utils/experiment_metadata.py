"""Save configs, commands and file paths for training and test runs.

Also record file hashes for later comparison.
"""

import hashlib
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path, chunk_size=1024 * 1024):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return repr(value)


def resolved_config(cfg, config_name=None):
    values = {
        name: _json_value(getattr(cfg, name))
        for name in sorted(dir(cfg))
        if name.isupper() and not callable(getattr(cfg, name))
    }
    return {
        "registry_name": config_name,
        "class_name": cfg.__name__,
        "module": cfg.__module__,
        "values": values,
    }


def _file_record(path):
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    record = {"path": str(resolved), "exists": resolved.is_file()}
    if resolved.is_file():
        record["size_bytes"] = resolved.stat().st_size
        record["sha256"] = sha256_file(resolved)
    return record


def _write_resolved_config(cfg, exp_dir, config_name):
    config_path = Path(exp_dir) / "config.json"
    with config_path.open("w") as f:
        json.dump(
            resolved_config(cfg, config_name=config_name),
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")
    return config_path


def _command_record(command):
    argv = [str(value) for value in command] if command is not None else []
    return {
        "argv": argv,
        "shell": shlex.join(argv) if argv else None,
    }


def write_train_provenance(
        cfg,
        exp_dir,
        config_name=None,
        command=None,
        status="running",
        checkpoint_path=None):
    """Write the effective configuration and metadata for a training run."""
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    config_path = _write_resolved_config(cfg, exp_dir, config_name)

    metadata = {
        "schema_version": 2,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "train",
        "status": status,
        "experiment_name": exp_dir.parent.name,
        "command": _command_record(command),
        "config": {
            "registry_name": config_name,
            "snapshot": config_path.name,
            "snapshot_sha256": sha256_file(config_path),
        },
        "checkpoint": _file_record(checkpoint_path),
        "data": {
            "train_annotation": _file_record(getattr(cfg, "TRAIN_ANN_FILE", None)),
            "validation_annotation": _file_record(getattr(cfg, "VAL_ANN_FILE", None)),
            "test_annotation": _file_record(getattr(cfg, "TEST_ANN_FILE", None)),
            "validation_occlusion_annotation": _file_record(
                getattr(cfg, "VAL_OCCLUSION_FILE", None)
            ),
            "validation_not_occluded_annotation": _file_record(
                getattr(cfg, "VAL_NOT_OCCLUDED_FILE", None)
            ),
            "test_occlusion_annotation": _file_record(
                getattr(cfg, "TEST_OCCLUSION_FILE", None)
            ),
            "test_not_occluded_annotation": _file_record(
                getattr(cfg, "TEST_NOT_OCCLUDED_FILE", None)
            ),
        },
    }

    metadata_path = exp_dir / "metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    return config_path, metadata_path


def write_test_provenance(
        cfg,
        exp_dir,
        checkpoint_path,
        config_name=None,
        command=None,
        test_annotation_path=None,
        test_occlusion_path=None,
        test_not_occluded_path=None,
        test_image_dir=None):
    """Write the effective config and metadata for a completed test run."""
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    config_path = _write_resolved_config(cfg, exp_dir, config_name)
    metadata = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "test",
        "experiment_name": exp_dir.parent.name,
        "command": _command_record(command),
        "config": {
            "registry_name": config_name,
            "snapshot": config_path.name,
            "snapshot_sha256": sha256_file(config_path),
        },
        "checkpoint": _file_record(checkpoint_path),
        "test_data": {
            "annotation": _file_record(test_annotation_path),
            "occlusion_annotation": _file_record(test_occlusion_path),
            "not_occluded_annotation": _file_record(test_not_occluded_path),
            "image_directory": (
                str(Path(test_image_dir).expanduser().resolve())
                if test_image_dir is not None else None
            ),
        },
    }

    metadata_path = exp_dir / "metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    return config_path, metadata_path
