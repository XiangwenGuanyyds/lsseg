"""Dice-weight-16 model with gated residual mask refinement."""

from utils.registry import CONFIGS

from .dice_w16_cfg import DiceW16Config


@CONFIGS.register_module(name="dice_w16_gated_residual_refinement")
class DiceW16GatedResidualRefinementConfig(DiceW16Config):
    MODEL = {
        **DiceW16Config.MODEL,
        "residual_mask_refinement": {
            "type": "ResidualMaskRefinementHead",
            "in_channels": 256,
            "hidden_channels": 64,
            "featmap_names": ("0",),
            "roi_output_size": 28,
            "sampling_ratio": 2,
            "output_size": 56,
            "gate_kernel_size": 3,
            "foreground_class": 1,
            "detach_coarse_context": True,
        },
    }
