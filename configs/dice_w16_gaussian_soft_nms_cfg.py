"""Dice-weight-16 checkpoint inference with standard Gaussian Soft-NMS."""

from utils.registry import CONFIGS

from .dice_w16_cfg import DiceW16Config


@CONFIGS.register_module(name="dice_w16_gaussian_soft_nms")
class DiceW16GaussianSoftNMSConfig(DiceW16Config):
    MODEL = {
        **DiceW16Config.MODEL,
        "box_nms": {
            "type": "soft_nms",
            "sigma": 0.5,
            "score_thresh_keep": 1e-3,
        },
    }
