"""Boundary-loss model with gated residual mask refinement."""

from utils.registry import CONFIGS

from .boundary_loss_w0_3_cfg import BoundaryLossW03Config


@CONFIGS.register_module(name="boundary_loss_w0_3_gated_residual_refinement")
class BoundaryLossW03GatedResidualRefinementConfig(BoundaryLossW03Config):
    MODEL = {
        **BoundaryLossW03Config.MODEL,
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
