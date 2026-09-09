"""Mixed-scene + BCE/Boundary mask loss with boundary weight 0.1."""

from utils.registry import CONFIGS

from .boundary_w03_cfg import BoundaryW03Config


@CONFIGS.register_module(name="boundary_w01")
class BoundaryW01Config(BoundaryW03Config):
    TASK_NAME = "boundary_w01"

    LOSS = {
        "type": "LossAggregator",
        "loss_weights": {
            "loss_mask_bce": 1.0,
            "loss_mask_boundary": 0.1,
        },
    }
