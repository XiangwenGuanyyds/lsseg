"""Mixed-scene Mask R-CNN with pooled neighboring-RoI mask correction."""

from utils.registry import CONFIGS

from .baseline_cfg import BaselineConfig


@CONFIGS.register_module(name="pooled_context_mask_correction")
class PooledContextMaskCorrectionConfig(BaselineConfig):
    TASK_NAME = "pooled_context_mask_correction"

    MODEL = {
        **BaselineConfig.MODEL,
        "mask_correction": {
            "type": "PooledContextMaskCorrection",
            "in_channels": 256,
            "context_dim": 64,
            "relation_hidden_dim": 64,
            "neighbor_top_k": 4,
            "neighbor_iou_threshold": 0.1,
            "delta_scale": 1.0,
            "foreground_class": 1,
        },
    }
