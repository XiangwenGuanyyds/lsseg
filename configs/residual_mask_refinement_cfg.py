"""Baseline with the final Residual Mask Refinement module."""

from utils.registry import CONFIGS

from .baseline_cfg import BaselineConfig


@CONFIGS.register_module(name="residual_mask_refinement")
class ResidualMaskRefinementConfig(BaselineConfig):
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
            "uncertainty_pool_kernel_size": 3,
            "foreground_class": 1,
            "detach_coarse_context": True,
            "high_resolution_encoder_mode": "conv1x1",
            "refinement_feature_channels": 64,
            "use_uncertainty_weight": True,
            "use_uncertainty_weight_in_predictor": False,
        },
    }
