"""Mixed-scene + BCE/Dice/Boundary mask loss.

Initial combination check:
    loss_mask_bce = 1.0
    loss_mask_dice = 2.0
    loss_mask_boundary = 0.5
"""

from utils.registry import CONFIGS

from .baseline_cfg import BaselineConfig


@CONFIGS.register_module(name="bce_dice_w20_boundary_w05")
class BCEDiceW20BoundaryW05Config(BaselineConfig):
    TASK_NAME = "bce_dice_w20_boundary_w05"

    MODEL = {
        **BaselineConfig.MODEL,
        "mask_loss": "bce_dice_boundary_loss",
    }

    LOSS = {
        "type": "LossAggregator",
        "loss_weights": {
            "loss_mask_bce": 1.0,
            "loss_mask_dice": 2.0,
            "loss_mask_boundary": 0.5,
        },
    }
