"""Auto-import every ``*_cfg.py`` in this directory.

Each config file registers its class via ``@CONFIGS.register_module(name=...)``
at import time. Importing this package therefore populates the CONFIGS
registry with every config available in the project — no hand-maintained
list anywhere.

Use:
    from utils.registry import CONFIGS
    import configs   # triggers auto-registration
    cfg_cls = CONFIGS.get("baseline")
"""

import importlib
import pkgutil
from pathlib import Path

_pkg_dir = Path(__file__).parent

# Import every sibling module ending in `_cfg.py`. Order doesn't matter for
# correctness — `from .x import Y` style inheritance handles itself — but we
# sort for deterministic load order so any registration error message is
# stable across runs.
for module_info in sorted(pkgutil.iter_modules([str(_pkg_dir)]),
                          key=lambda m: m.name):
    if module_info.name.endswith("_cfg"):
        importlib.import_module(f"{__name__}.{module_info.name}")

del importlib, pkgutil, Path, _pkg_dir, module_info
