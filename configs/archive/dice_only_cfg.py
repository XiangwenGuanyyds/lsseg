"""Mixed-scene + Dice-only mask loss.

This config keeps the same FPN + Mask R-CNN architecture and data settings as
the baseline, but trains the mask branch with Dice loss only.

Run:
    python main.py --mode train --config dice_only --exp_name <name>
    python main.py --mode test  --config dice_only --exp_name <name>
"""

from utils.registry import CONFIGS

from .baseline_cfg import BaselineConfig


@CONFIGS.register_module(name="dice_only")
class DiceOnlyConfig(BaselineConfig):
    TASK_NAME = "dice_only"

    MODEL = {
        **BaselineConfig.MODEL,
        "mask_loss": "dice_loss",
    }

    LOSS = {
        "type": "LossAggregator",
        "loss_weights": {
            "loss_mask_dice": 1.0,
        },
    }
