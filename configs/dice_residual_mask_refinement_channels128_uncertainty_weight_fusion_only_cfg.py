"""Dice and 128-channel refinement with fusion-only uncertainty weighting."""

from utils.registry import CONFIGS

from .dice_w16_mask_refinement_feature_channels_128_cfg import (
    DiceW16MaskRefinementFeatureChannels128Config,
)


@CONFIGS.register_module(
    name=(
        "dice_residual_mask_refinement_channels128_"
        "uncertainty_weight_fusion_only"
    )
)
class DiceResidualMaskRefinementChannels128UncertaintyWeightFusionOnlyConfig(
    DiceW16MaskRefinementFeatureChannels128Config
):
    MODEL = {
        **DiceW16MaskRefinementFeatureChannels128Config.MODEL,
        "residual_mask_refinement": {
            **DiceW16MaskRefinementFeatureChannels128Config.MODEL[
                "residual_mask_refinement"
            ],
            "use_spatial_weight": True,
            "use_spatial_weight_in_predictor": False,
        },
    }
