"""Mixed-scene baseline + CBAM feature attention.

This config is the clean attention-only ablation against baseline:
same data, loss, optimizer, NMS, and diagnostics; the only model change is
applying CBAM to each FPN feature map after the backbone and before RPN/ROI
heads.

Run:
    python main.py --mode train --config cbam --exp_name <name>
    python main.py --mode test  --config cbam --exp_name <name>
"""

from utils.registry import CONFIGS

from .baseline_cfg import BaselineConfig


@CONFIGS.register_module(name="cbam")
class CBAMConfig(BaselineConfig):
    TASK_NAME = "cbam"

    MODEL = {
        **BaselineConfig.MODEL,
        "attention": {
            "type": "CBAM",
            "in_channels": 256,
            "reduction": 16,
            "spatial_kernel": 7,
        },
    }
