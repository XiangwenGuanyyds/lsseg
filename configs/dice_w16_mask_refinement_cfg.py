"""Final Dice-w16 Mask Refinement configuration."""

from utils.registry import CONFIGS

from .dice_w16_mask_refinement_feature_channels_64_cfg import (
    DiceW16MaskRefinementFeatureChannels64Config,
)


@CONFIGS.register_module(name="dice_w16_mask_refinement")
class DiceW16MaskRefinementConfig(
    DiceW16MaskRefinementFeatureChannels64Config
):
    MODEL = {
        **DiceW16MaskRefinementFeatureChannels64Config.MODEL,
        "residual_mask_refinement": {
            **DiceW16MaskRefinementFeatureChannels64Config.MODEL[
                "residual_mask_refinement"
            ],
            "high_resolution_encoder_mode": "channel_projection",
            "refinement_feature_channels": 64,
            "use_spatial_weight": True,
            "use_spatial_weight_in_predictor": True,
        },
    }
