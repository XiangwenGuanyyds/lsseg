"""Train Dice and Residual Mask Refinement with standard NMS for validation."""

from utils.registry import CONFIGS
from .dice_cfg import DiceConfig
from .residual_mask_refinement_cfg import ResidualMaskRefinementConfig


@CONFIGS.register_module(name="dice_residual_mask_refinement")
class DiceResidualMaskRefinementConfig(DiceConfig):
    MODEL = {
        **DiceConfig.MODEL,
        "residual_mask_refinement": {
            **ResidualMaskRefinementConfig.MODEL["residual_mask_refinement"],
        },
    }
