"""Mixed-scene + BCE/Dice mask loss with Dice weight 2.0.

Used for the seed0 loss-weight ablation. It inherits the BCE+Dice config and
only changes the Dice loss weight.

Run:
    python main.py --mode train --config bce_dice_w20 --exp_name <name>
    python main.py --mode test  --config bce_dice_w20 --exp_name <name>
"""

from utils.registry import CONFIGS

from .bce_dice_cfg import BCEDiceConfig


@CONFIGS.register_module(name="bce_dice_w20")
class BCEDiceW20Config(BCEDiceConfig):
    TASK_NAME = "bce_dice_w20"

    LOSS = {
        "type": "LossAggregator",
        "loss_weights": {
            "loss_mask_bce": 1.0,
            "loss_mask_dice": 2.0,
        },
    }
