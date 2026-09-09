"""Mixed-scene + BCE/Dice mask loss with Dice weight 128.0."""

from utils.registry import CONFIGS

from .bce_dice_cfg import BCEDiceConfig


@CONFIGS.register_module(name="bce_dice_w1280")
class BCEDiceW1280Config(BCEDiceConfig):
    TASK_NAME = "bce_dice_w1280"

    LOSS = {
        "type": "LossAggregator",
        "loss_weights": {
            "loss_mask_bce": 1.0,
            "loss_mask_dice": 128.0,
        },
    }
