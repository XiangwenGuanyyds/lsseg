"""Dice-weight-16 checkpoint inference with thresholded Gaussian Soft-NMS."""

from utils.registry import CONFIGS

from .dice_w16_cfg import DiceW16Config


@CONFIGS.register_module(name="dice_w16_thresholded_gaussian_soft_nms")
class DiceW16ThresholdedGaussianSoftNMSConfig(DiceW16Config):
    MODEL = {
        **DiceW16Config.MODEL,
        "box_nms": {
            "type": "thresholded_gaussian_soft_nms",
            "sigma": 0.5,
            "score_thresh_keep": 1e-3,
        },
    }
