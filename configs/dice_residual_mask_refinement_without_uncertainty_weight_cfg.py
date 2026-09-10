"""Train Dice and Residual Mask Refinement without uncertainty weighting."""

from utils.registry import CONFIGS
from .dice_residual_mask_refinement_cfg import DiceResidualMaskRefinementConfig


@CONFIGS.register_module(name="dice_residual_mask_refinement_without_uncertainty_weight")
class DiceResidualMaskRefinementWithoutUncertaintyWeightConfig(DiceResidualMaskRefinementConfig):
    MODEL = {
        **DiceResidualMaskRefinementConfig.MODEL,
        "residual_mask_refinement": {
            **DiceResidualMaskRefinementConfig.MODEL["residual_mask_refinement"],
            "use_uncertainty_weight": False,
        },
    }
