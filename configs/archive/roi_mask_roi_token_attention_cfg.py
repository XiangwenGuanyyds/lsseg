"""Mixed-scene Mask R-CNN with one self-attention token per RoI.

Run:
    python main.py --mode train --config roi_mask_roi_token_attention --exp_name <name>
    python main.py --mode test  --config roi_mask_roi_token_attention --exp_name <name>
"""

from utils.registry import CONFIGS

from .baseline_cfg import BaselineConfig


@CONFIGS.register_module(name="roi_mask_roi_token_attention")
class RoIMaskRoITokenAttentionConfig(BaselineConfig):
    TASK_NAME = "roi_mask_roi_token_attention"

    MODEL = {
        **BaselineConfig.MODEL,
        "mask_attention": {
            "type": "RoIMaskSelfAttention",
            "in_channels": 256,
            "token_height": 14,
            "token_width": 14,
            "token_dim": 256,
            "num_heads": 4,
            "dropout": 0.0,
            "attention_scale": 1.0,
        },
    }
