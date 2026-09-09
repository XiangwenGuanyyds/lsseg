"""Mixed-scene + BCE/Dice mask loss + Soft-NMS at test time.

This config inherits the trained-model side from bce_dice and only
changes the box-head NMS function. Soft-NMS is a post-processing change, so it
can evaluate existing BCE+Dice checkpoints via --weights without retraining.

Run:
    ./lsseg test bce_dice_seed0 --config bce_dice_soft_nms \
        --result-name bce_dice_soft_nms_seed0
"""

from utils.registry import CONFIGS

from .bce_dice_cfg import BCEDiceConfig


@CONFIGS.register_module(name="bce_dice_soft_nms")
class BCEDiceSoftNMSConfig(BCEDiceConfig):
    TASK_NAME = "bce_dice_soft_nms"

    MODEL = {
        **BCEDiceConfig.MODEL,
        "box_nms": {
            "type":              "soft_nms",
            # Standard Gaussian Soft-NMS settings from Bodla et al. (2017).
            "sigma":             0.5,
            "score_thresh_keep": 1e-3,
        },
    }
