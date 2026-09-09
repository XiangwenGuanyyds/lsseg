"""Dice-w16 mask refinement with spatial weighting only during fusion."""

from utils.registry import CONFIGS

from .dice_w16_mask_refinement_feature_channels_64_cfg import (
    DiceW16MaskRefinementFeatureChannels64Config,
)


@CONFIGS.register_module(
    name=(
        "dice_w16_mask_refinement_feature_channels_64_"
        "spatial_weight_fusion_only"
    )
)
class DiceW16MaskRefinementFeatureChannels64SpatialWeightFusionOnlyConfig(
    DiceW16MaskRefinementFeatureChannels64Config
):
    MODEL = {
        **DiceW16MaskRefinementFeatureChannels64Config.MODEL,
        "residual_mask_refinement": {
            **DiceW16MaskRefinementFeatureChannels64Config.MODEL[
                "residual_mask_refinement"
            ],
            "use_spatial_weight": True,
            "use_spatial_weight_in_predictor": False,
        },
    }
