"""Mixed-scene baseline + RoI mask self-attention.

This config isolates the RoI-level mask attention module against
baseline. The module is inserted after mask_roi_pool and before
mask_head; data, loss, optimizer, and NMS follow the baseline config.

Run:
    python main.py --mode train --config roi_mask_self_attention --exp_name <name>
    python main.py --mode test  --config roi_mask_self_attention --exp_name <name>
"""

from utils.registry import CONFIGS

from .baseline_cfg import BaselineConfig


@CONFIGS.register_module(name="roi_mask_self_attention")
class RoIMaskSelfAttentionConfig(BaselineConfig):
    TASK_NAME = "roi_mask_self_attention"

    MODEL = {
        **BaselineConfig.MODEL,
        "mask_attention": {
            "type": "RoIMaskSelfAttention",
            "in_channels": 256,
            "num_heads": 4,
            "dropout": 0.0,
            "attention_scale": 1.0,
        },
    }
