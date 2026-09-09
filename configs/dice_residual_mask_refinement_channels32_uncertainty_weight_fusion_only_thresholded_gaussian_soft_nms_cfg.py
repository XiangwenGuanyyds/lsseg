"""Test-time NMS for 32-channel fusion-only uncertainty refinement."""

from utils.registry import CONFIGS

from .dice_residual_mask_refinement_channels32_uncertainty_weight_fusion_only_cfg import (
    DiceResidualMaskRefinementChannels32UncertaintyWeightFusionOnlyConfig,
)


@CONFIGS.register_module(
    name=(
        "dice_residual_mask_refinement_channels32_"
        "uncertainty_weight_fusion_only_thresholded_gaussian_soft_nms"
    )
)
class DiceResidualMaskRefinementChannels32UncertaintyWeightFusionOnlyThresholdedGaussianSoftNMSConfig(
    DiceResidualMaskRefinementChannels32UncertaintyWeightFusionOnlyConfig
):
    MODEL = {
        **DiceResidualMaskRefinementChannels32UncertaintyWeightFusionOnlyConfig.MODEL,
        "box_nms": {
            "type": "thresholded_gaussian_soft_nms",
            "sigma": 0.5,
            "score_thresh_keep": 1e-3,
        },
    }
