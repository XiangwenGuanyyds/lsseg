"""Dice-w16 Mask Refinement with boundary supervision at weight 0.3."""

from utils.registry import CONFIGS

from .dice_w16_gated_residual_refinement_cfg import (
    DiceW16GatedResidualRefinementConfig,
)


@CONFIGS.register_module(
    name="dice_w16_mask_refinement_boundary_loss_w0_3"
)
class DiceW16MaskRefinementBoundaryLossW03Config(
    DiceW16GatedResidualRefinementConfig
):
    MODEL = {
        **DiceW16GatedResidualRefinementConfig.MODEL,
        "mask_loss": "bce_dice_boundary_loss",
    }
    LOSS = {
        "type": "LossAggregator",
        "loss_weights": {
            "loss_mask_bce": 1.0,
            "loss_mask_dice": 16.0,
            "loss_mask_boundary": 0.3,
        },
    }
