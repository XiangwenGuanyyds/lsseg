"""Baseline inference with the earlier thresholded Gaussian NMS variant.

The baseline checkpoint is reused without retraining. Gaussian score decay is
applied only when the IoU exceeds 0.5, matching the earlier project behavior.

Run:
    ./lsseg test baseline_seed0 --config thresholded_soft_nms \
        --result-name thresholded_soft_nms_seed0
"""

from utils.registry import CONFIGS

from .baseline_cfg import BaselineConfig


@CONFIGS.register_module(name="thresholded_soft_nms")
class ThresholdedSoftNMSConfig(BaselineConfig):
    TASK_NAME = "thresholded_soft_nms"

    MODEL = {
        **BaselineConfig.MODEL,
        "box_nms": {
            "type": "thresholded_gaussian_soft_nms",
            "sigma": 0.5,
            "score_thresh_keep": 1e-3,
        },
    }
