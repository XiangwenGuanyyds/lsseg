from datetime import datetime

from utils.output_paths import experiment_paths


def get_experiment_dir(cfg):
    """Resolve and create ``outputs/<experiment>/training/``.

    Falls back to a timestamp when ``cfg.EXP_NAME`` is not set so unnamed
    runs still get an isolated directory instead of overwriting each other.
    """
    exp_name = getattr(cfg, "EXP_NAME", None)
    if not exp_name:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_dir = experiment_paths(cfg.BASE_DIR, exp_name).training
    exp_dir.mkdir(parents=True, exist_ok=True)
    return str(exp_dir), exp_name
