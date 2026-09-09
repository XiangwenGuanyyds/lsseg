"""Baseline with CBAM applied to the FPN feature maps."""

from utils.registry import CONFIGS

from .baseline_cfg import BaselineConfig


@CONFIGS.register_module(name="cbam")
class CBAMConfig(BaselineConfig):
    MODEL = {
        **BaselineConfig.MODEL,
        "attention": {
            "type": "CBAM",
            "in_channels": 256,
            "reduction": 16,
            "spatial_kernel": 7,
        },
    }
