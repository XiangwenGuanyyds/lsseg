"""Test the Dice + Residual Mask Refinement model with Thresholded Gaussian Soft-NMS."""

from utils.registry import CONFIGS
from .dice_residual_mask_refinement_cfg import DiceResidualMaskRefinementConfig
from .thresholded_gaussian_soft_nms_cfg import ThresholdedGaussianSoftNMSConfig


@CONFIGS.register_module(name="final_model")
class FinalModelConfig(DiceResidualMaskRefinementConfig):
    TEST_ONLY = True
    MODEL = {
        **DiceResidualMaskRefinementConfig.MODEL,
        "box_nms": {**ThresholdedGaussianSoftNMSConfig.MODEL["box_nms"]},
    }
