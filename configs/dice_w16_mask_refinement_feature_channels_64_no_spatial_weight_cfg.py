"""Dice-w16 mask refinement without the spatial uncertainty weight."""

from utils.registry import CONFIGS

from .dice_w16_mask_refinement_feature_channels_64_cfg import (
    DiceW16MaskRefinementFeatureChannels64Config,
)


@CONFIGS.register_module(
    name="dice_w16_mask_refinement_feature_channels_64_no_spatial_weight"
)
class DiceW16MaskRefinementFeatureChannels64NoSpatialWeightConfig(
    DiceW16MaskRefinementFeatureChannels64Config
):
    MODEL = {
        **DiceW16MaskRefinementFeatureChannels64Config.MODEL,
        "residual_mask_refinement": {
            **DiceW16MaskRefinementFeatureChannels64Config.MODEL[
                "residual_mask_refinement"
            ],
            "use_spatial_weight": False,
        },
    }
