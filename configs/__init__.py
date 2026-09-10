"""Register the public experiment configs in this directory."""

import importlib
import pkgutil
from pathlib import Path

_pkg_dir = Path(__file__).parent

# Archive subdirectories are excluded.
for module_info in sorted(pkgutil.iter_modules([str(_pkg_dir)]),
                          key=lambda m: m.name):
    if module_info.name.endswith("_cfg"):
        importlib.import_module(f"{__name__}.{module_info.name}")

del importlib, pkgutil, Path, _pkg_dir, module_info
