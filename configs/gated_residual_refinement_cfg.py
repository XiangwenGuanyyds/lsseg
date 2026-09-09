"""Baseline with gated residual mask refinement."""

from utils.registry import CONFIGS

from .baseline_cfg import BaselineConfig


@CONFIGS.register_module(name="gated_residual_refinement")
class GatedResidualRefinementConfig(BaselineConfig):
    MODEL = {
        **BaselineConfig.MODEL,
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
