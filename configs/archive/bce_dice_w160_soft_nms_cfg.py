"""Mixed-scene + BCE/Dice weight 16.0 + Soft-NMS at test time.

Soft-NMS is a post-processing change. This config is intended to evaluate
existing BCE+Dice w16.0 checkpoints via --weights without retraining.

Run:
    ./lsseg test dice_w16_seed0 --config bce_dice_w160_soft_nms \
        --result-name dice_w16_soft_nms_seed0
"""

from utils.registry import CONFIGS

from .bce_dice_w160_cfg import BCEDiceW160Config


@CONFIGS.register_module(name="bce_dice_w160_soft_nms")
class BCEDiceW160SoftNMSConfig(BCEDiceW160Config):
    TASK_NAME = "bce_dice_w160_soft_nms"

    MODEL = {
        **BCEDiceW160Config.MODEL,
        "box_nms": {
            "type":              "soft_nms",
            # Standard Gaussian Soft-NMS settings from Bodla et al. (2017).
            "sigma":             0.5,
            "score_thresh_keep": 1e-3,
        },
    }
