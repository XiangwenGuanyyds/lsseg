"""BCE+Dice w16 with an uncertainty-gated residual mask refinement head.

The detector, standard NMS, and existing BCE+Dice loss remain unchanged. The
optional head refines 28x28 coarse foreground logits to 56x56 using P2 RoI
features. Configs without ``MODEL['residual_mask_refinement']`` keep the
original Mask R-CNN mask path.
"""

from utils.registry import CONFIGS

from .bce_dice_w160_cfg import BCEDiceW160Config


@CONFIGS.register_module(
    name="bce_dice_w160_gated_residual_refinement"
)
class BCEDiceW160GatedResidualRefinementConfig(
    BCEDiceW160Config
):
    TASK_NAME = "bce_dice_w160_gated_residual_refinement"

    MODEL = {
        **BCEDiceW160Config.MODEL,
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
