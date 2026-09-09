"""Dice and Residual Mask Refinement with fusion-only uncertainty weighting."""

from utils.registry import CONFIGS

from .dice_residual_mask_refinement_uncertainty_weight_fusion_only_cfg import (
    DiceResidualMaskRefinementUncertaintyWeightFusionOnlyConfig,
)


@CONFIGS.register_module(
    name=(
        "dice_residual_mask_refinement_uncertainty_weight_fusion_only_"
        "thresholded_gaussian_soft_nms"
    )
)
class DiceResidualMaskRefinementUncertaintyWeightFusionOnlyThresholdedGaussianSoftNMSConfig(
    DiceResidualMaskRefinementUncertaintyWeightFusionOnlyConfig
):
    MODEL = {
        **DiceResidualMaskRefinementUncertaintyWeightFusionOnlyConfig.MODEL,
        "box_nms": {
            "type": "thresholded_gaussian_soft_nms",
            "sigma": 0.5,
            "score_thresh_keep": 1e-3,
        },
    }
