"""Test the no-weight refinement model with Thresholded Gaussian Soft-NMS."""

from utils.registry import CONFIGS
from .dice_residual_mask_refinement_without_uncertainty_weight_cfg import (
    DiceResidualMaskRefinementWithoutUncertaintyWeightConfig,
)
from .thresholded_gaussian_soft_nms_cfg import ThresholdedGaussianSoftNMSConfig


@CONFIGS.register_module(name="final_model_without_uncertainty_weight")
class FinalModelWithoutUncertaintyWeightConfig(DiceResidualMaskRefinementWithoutUncertaintyWeightConfig):
    TEST_ONLY = True
    MODEL = {
        **DiceResidualMaskRefinementWithoutUncertaintyWeightConfig.MODEL,
        "box_nms": {**ThresholdedGaussianSoftNMSConfig.MODEL["box_nms"]},
    }
