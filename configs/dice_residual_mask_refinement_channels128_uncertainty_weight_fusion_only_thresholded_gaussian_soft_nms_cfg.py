"""Test-time NMS for 128-channel fusion-only uncertainty refinement."""

from utils.registry import CONFIGS

from .dice_residual_mask_refinement_channels128_uncertainty_weight_fusion_only_cfg import (
    DiceResidualMaskRefinementChannels128UncertaintyWeightFusionOnlyConfig,
)


@CONFIGS.register_module(
    name=(
        "dice_residual_mask_refinement_channels128_"
        "uncertainty_weight_fusion_only_thresholded_gaussian_soft_nms"
    )
)
class DiceResidualMaskRefinementChannels128UncertaintyWeightFusionOnlyThresholdedGaussianSoftNMSConfig(
    DiceResidualMaskRefinementChannels128UncertaintyWeightFusionOnlyConfig
):
    MODEL = {
        **DiceResidualMaskRefinementChannels128UncertaintyWeightFusionOnlyConfig.MODEL,
        "box_nms": {
            "type": "thresholded_gaussian_soft_nms",
            "sigma": 0.5,
            "score_thresh_keep": 1e-3,
        },
    }
