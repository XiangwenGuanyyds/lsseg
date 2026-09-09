"""Baseline inference with Gaussian Soft-NMS.

Same as baseline except cfg.MODEL["box_nms"] is set to soft_nms.
The trained baseline checkpoint is reused; no retraining is required.

Run:
    ./lsseg test baseline_seed0 --config soft_nms \
        --result-name soft_nms_seed0
"""

from utils.registry import CONFIGS

from .baseline_cfg import BaselineConfig


@CONFIGS.register_module(name="soft_nms")
class SoftNMSConfig(BaselineConfig):
    TASK_NAME = "soft_nms"

    MODEL = {
        **BaselineConfig.MODEL,
        "box_nms": {
            "type":              "soft_nms",
            # Standard Gaussian Soft-NMS settings from Bodla et al. (2017).
            "sigma":             0.5,
            "score_thresh_keep": 1e-3,
        },
    }
