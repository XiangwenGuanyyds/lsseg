"""Baseline with BCE and Dice mask supervision at Dice weight 16."""

from utils.registry import CONFIGS

from .baseline_cfg import BaselineConfig


@CONFIGS.register_module(name="dice_w16")
class DiceW16Config(BaselineConfig):
    MODEL = {
        **BaselineConfig.MODEL,
        "mask_loss": "bce_dice_loss",
    }
    LOSS = {
        "type": "LossAggregator",
        "loss_weights": {
            "loss_mask_bce": 1.0,
            "loss_mask_dice": 16.0,
        },
    }
