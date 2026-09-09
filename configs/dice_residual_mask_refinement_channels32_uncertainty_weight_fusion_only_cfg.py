"""Dice and 32-channel refinement with fusion-only uncertainty weighting."""

from utils.registry import CONFIGS

from .dice_w16_mask_refinement_feature_channels_32_cfg import (
    DiceW16MaskRefinementFeatureChannels32Config,
)


@CONFIGS.register_module(
    name=(
        "dice_residual_mask_refinement_channels32_"
        "uncertainty_weight_fusion_only"
    )
)
class DiceResidualMaskRefinementChannels32UncertaintyWeightFusionOnlyConfig(
    DiceW16MaskRefinementFeatureChannels32Config
):
    MODEL = {
        **DiceW16MaskRefinementFeatureChannels32Config.MODEL,
        "residual_mask_refinement": {
            **DiceW16MaskRefinementFeatureChannels32Config.MODEL[
                "residual_mask_refinement"
            ],
            "use_spatial_weight": True,
            "use_spatial_weight_in_predictor": False,
        },
    }
