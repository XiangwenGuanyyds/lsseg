"""Baseline inference with standard Gaussian Soft-NMS."""

from utils.registry import CONFIGS

from .baseline_cfg import BaselineConfig


@CONFIGS.register_module(name="gaussian_soft_nms")
class GaussianSoftNMSConfig(BaselineConfig):
    MODEL = {
        **BaselineConfig.MODEL,
        "box_nms": {
            "type": "soft_nms",
            "sigma": 0.5,
            "score_thresh_keep": 1e-3,
        },
    }
