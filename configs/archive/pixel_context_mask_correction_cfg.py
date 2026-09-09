"""Mixed-scene Mask R-CNN with pixel-level neighboring-RoI mask correction."""

from utils.registry import CONFIGS

from .baseline_cfg import BaselineConfig


@CONFIGS.register_module(name="pixel_context_mask_correction")
class PixelContextMaskCorrectionConfig(BaselineConfig):
    TASK_NAME = "pixel_context_mask_correction"

    MODEL = {
        **BaselineConfig.MODEL,
        "mask_correction": {
            "type": "PixelContextMaskCorrection",
            "in_channels": 256,
            "token_dim": 128,
            "num_heads": 4,
            "neighbor_top_k": 4,
            "neighbor_iou_threshold": 0.1,
            "relative_hidden_dim": 64,
            "delta_scale": 1.0,
            "foreground_class": 1,
        },
    }
