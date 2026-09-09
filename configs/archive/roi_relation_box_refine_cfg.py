"""Mixed-scene baseline + RoI relation box feature refinement.

This config adds a RoI relation module before the box head flattens RoIAlign
features. The module uses proposal geometry and neighbouring RoI features from
the same image. RPN, mask branch, mask loss, box regression loss, NMS,
optimizer, data loading, and augmentation follow the baseline config.

Run:
    python main.py --mode train --config roi_relation_box_refine --exp_name <name>
    python main.py --mode test  --config roi_relation_box_refine --exp_name <name>
"""

from utils.registry import CONFIGS

from .baseline_cfg import BaselineConfig


@CONFIGS.register_module(name="roi_relation_box_refine")
class RoIRelationBoxRefineConfig(BaselineConfig):
    TASK_NAME = "roi_relation_box_refine"

    MODEL = {
        **BaselineConfig.MODEL,
        "box_roi_relation_refinement": {
            "type": "RoIRelationBoxFeatureRefinement",
            "in_channels": 256,
            "relation_dim": 128,
            "geom_hidden_dim": 64,
            "top_k": 32,
            "residual_init": 0.1,
        },
    }
