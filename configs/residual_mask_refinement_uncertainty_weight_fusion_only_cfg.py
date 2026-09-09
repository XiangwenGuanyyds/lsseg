"""Baseline with uncertainty weighting only during residual fusion."""

from utils.registry import CONFIGS

from .residual_mask_refinement_cfg import ResidualMaskRefinementConfig


@CONFIGS.register_module(
    name="residual_mask_refinement_uncertainty_weight_fusion_only"
)
class ResidualMaskRefinementUncertaintyWeightFusionOnlyConfig(
    ResidualMaskRefinementConfig
):
    MODEL = {
        **ResidualMaskRefinementConfig.MODEL,
        "residual_mask_refinement": {
            **ResidualMaskRefinementConfig.MODEL["residual_mask_refinement"],
            "use_spatial_weight": True,
            "use_spatial_weight_in_predictor": False,
        },
    }
