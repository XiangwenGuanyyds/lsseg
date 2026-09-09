"""Mixed-scene baseline + overlap-aware box refinement.

This config isolates a structural change in the RoI box branch. The module
encodes proposal-to-proposal overlap statistics from the same image and injects
them only into the box regression feature. RPN, mask branch, mask loss, box
regression loss, NMS, optimizer, data loading, and augmentation follow the
baseline config.

Run:
    python main.py --mode train --config overlap_box_refine --exp_name <name>
    python main.py --mode test  --config overlap_box_refine --exp_name <name>
"""

from utils.registry import CONFIGS

from .baseline_cfg import BaselineConfig


@CONFIGS.register_module(name="overlap_box_refine")
class OverlapBoxRefineConfig(BaselineConfig):
    TASK_NAME = "overlap_box_refine"

    MODEL = {
        **BaselineConfig.MODEL,
        "box_overlap_refinement": {
            "type": "OverlapAwareBoxRefinement",
            "feature_dim": 1024,
            "context_hidden_dim": 128,
            "thresholds": (0.5, 0.7),
            "residual_init": 0.0,
        },
    }
