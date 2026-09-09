"""Dice-w16 mask refinement using 64-channel high-resolution features."""

from utils.registry import CONFIGS

from .dice_w16_mask_refinement_channel_projection_cfg import (
    DiceW16MaskRefinementChannelProjectionConfig,
)


@CONFIGS.register_module(name="dice_w16_mask_refinement_feature_channels_64")
class DiceW16MaskRefinementFeatureChannels64Config(
    DiceW16MaskRefinementChannelProjectionConfig
):
    MODEL = {
        **DiceW16MaskRefinementChannelProjectionConfig.MODEL,
        "residual_mask_refinement": {
            **DiceW16MaskRefinementChannelProjectionConfig.MODEL[
                "residual_mask_refinement"
            ],
            "refinement_feature_channels": 64,
        },
    }
