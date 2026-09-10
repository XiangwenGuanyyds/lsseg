"""Baseline inference with Thresholded Gaussian Soft-NMS."""

from utils.registry import CONFIGS

from .baseline_cfg import BaselineConfig


@CONFIGS.register_module(name="thresholded_gaussian_soft_nms")
class ThresholdedGaussianSoftNMSConfig(BaselineConfig):
    TEST_ONLY = True
    MODEL = {
        **BaselineConfig.MODEL,
        "box_nms": {
            "type": "thresholded_gaussian_soft_nms",
            "nms_thresh": 0.5,
            "sigma": 0.5,
            "score_thresh_keep": 1e-3,
        },
    }
