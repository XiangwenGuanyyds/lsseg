"""Baseline with the final Residual Mask Refinement module."""

from utils.registry import CONFIGS

from .gated_residual_refinement_cfg import GatedResidualRefinementConfig


@CONFIGS.register_module(name="residual_mask_refinement")
class ResidualMaskRefinementConfig(GatedResidualRefinementConfig):
    MODEL = {
        **GatedResidualRefinementConfig.MODEL,
        "residual_mask_refinement": {
            **GatedResidualRefinementConfig.MODEL[
                "residual_mask_refinement"
            ],
            "high_resolution_encoder_mode": "channel_projection",
            "refinement_feature_channels": 64,
            "use_spatial_weight": True,
            "use_spatial_weight_in_predictor": True,
        },
    }
