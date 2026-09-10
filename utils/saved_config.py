"""Read the configuration saved with an experiment."""

import json
from pathlib import Path
from types import SimpleNamespace


def load_saved_config(path, root):
    path = Path(path)
    saved = json.loads(path.read_text())
    values = saved.get("values")
    if not isinstance(values, dict) or not isinstance(values.get("MODEL"), dict):
        raise ValueError(f"Saved config has no MODEL settings: {path}")
    if not values["MODEL"].get("type"):
        raise ValueError(f"Saved config has no model type: {path}")
    root = Path(root).resolve()
    old_root = Path(values.get("BASE_DIR", root))
    for key, value in values.items():
        if isinstance(value, str) and (key.endswith("_DIR") or key.endswith("_FILE")):
            location = Path(value)
            if location.is_absolute():
                try:
                    location = root / location.relative_to(old_root)
                except ValueError:
                    pass
            else:
                location = root / location
            values[key] = str(location)
    values["BASE_DIR"] = str(root)
    if values.get("CATEGORY_REMAP"):
        values["CATEGORY_REMAP"] = {
            int(key): int(value) for key, value in values["CATEGORY_REMAP"].items()
        }
    cfg = SimpleNamespace(**values)
    cfg.__name__ = saved.get("class_name", "SavedConfig")
    cfg.__module__ = saved.get("module", "saved_config")
    return cfg, saved.get("registry_name")
