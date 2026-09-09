"""Dice-w16 mask refinement using only P2 channel projection."""

from utils.registry import CONFIGS

from .dice_w16_gated_residual_refinement_cfg import (
    DiceW16GatedResidualRefinementConfig,
)


@CONFIGS.register_module(name="dice_w16_mask_refinement_channel_projection")
class DiceW16MaskRefinementChannelProjectionConfig(
    DiceW16GatedResidualRefinementConfig
):
    MODEL = {
        **DiceW16GatedResidualRefinementConfig.MODEL,
        "residual_mask_refinement": {
            **DiceW16GatedResidualRefinementConfig.MODEL[
                "residual_mask_refinement"
            ],
            "high_resolution_encoder_mode": "channel_projection",
        },
    }
