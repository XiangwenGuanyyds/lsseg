"""Baseline inference with Gaussian score decay above IoU 0.5."""

from utils.registry import CONFIGS

from .thresholded_gaussian_soft_nms_cfg import (
    ThresholdedGaussianSoftNMSConfig,
)


@CONFIGS.register_module(name="thresholded_gaussian_soft_nms_iou0_5")
class ThresholdedGaussianSoftNMSIoU05Config(
    ThresholdedGaussianSoftNMSConfig
):
    MODEL = {
        **ThresholdedGaussianSoftNMSConfig.MODEL,
        "box_nms": {
            **ThresholdedGaussianSoftNMSConfig.MODEL["box_nms"],
            "nms_thresh": 0.5,
        },
    }
