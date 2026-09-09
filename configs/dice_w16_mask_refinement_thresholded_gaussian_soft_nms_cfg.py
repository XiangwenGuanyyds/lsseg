"""Dice-w16 Mask Refinement inference with Thresholded Gaussian Soft-NMS."""

from utils.registry import CONFIGS

from .dice_w16_mask_refinement_cfg import (
    DiceW16MaskRefinementConfig,
)


@CONFIGS.register_module(
    name="dice_w16_mask_refinement_thresholded_gaussian_soft_nms"
)
class DiceW16MaskRefinementThresholdedGaussianSoftNMSConfig(
    DiceW16MaskRefinementConfig
):
    MODEL = {
        **DiceW16MaskRefinementConfig.MODEL,
        "box_nms": {
            "type": "thresholded_gaussian_soft_nms",
            "sigma": 0.5,
            "score_thresh_keep": 1e-3,
        },
    }
