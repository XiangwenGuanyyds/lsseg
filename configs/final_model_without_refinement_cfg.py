"""Test the Dice model with Thresholded Gaussian Soft-NMS; refinement is removed."""

from utils.registry import CONFIGS
from .dice_cfg import DiceConfig
from .thresholded_gaussian_soft_nms_cfg import ThresholdedGaussianSoftNMSConfig


@CONFIGS.register_module(name="final_model_without_refinement")
class FinalModelWithoutRefinementConfig(DiceConfig):
    TEST_ONLY = True
    MODEL = {
        **DiceConfig.MODEL,
        "box_nms": {**ThresholdedGaussianSoftNMSConfig.MODEL["box_nms"]},
    }
