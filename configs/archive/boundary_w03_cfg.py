"""Mixed-scene + BCE/Boundary mask loss with boundary weight 0.3.

This is a lightweight reproduction of the earlier boundary-loss attempt under
the current scene_mix experiment setup. It keeps the model architecture
unchanged and only swaps the mask loss through the MASK_LOSSES registry.

Run:
    python main.py --mode train --config boundary_w03 --exp_name <name>
    python main.py --mode test  --config boundary_w03 --exp_name <name>
"""

from utils.registry import CONFIGS

from .baseline_cfg import BaselineConfig


@CONFIGS.register_module(name="boundary_w03")
class BoundaryW03Config(BaselineConfig):
    TASK_NAME = "boundary_w03"

    MODEL = {
        **BaselineConfig.MODEL,
        "mask_loss": "bce_boundary_loss",
    }

    LOSS = {
        "type": "LossAggregator",
        "loss_weights": {
            "loss_mask_bce": 1.0,
            "loss_mask_boundary": 0.3,
        },
    }
