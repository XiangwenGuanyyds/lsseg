"""Mixed-scene RoI relation box refinement with auxiliary edge loss.

This config uses the RoI relation module before the box head flattening step
and adds a light same-instance edge loss during training. The auxiliary loss
uses training-time RoI-to-GT matches and is disabled during inference.

Run:
    python main.py --mode train --config roi_relation_box_refine_edge_loss --exp_name <name>
    python main.py --mode test  --config roi_relation_box_refine_edge_loss --exp_name <name>
"""

from utils.registry import CONFIGS

from .baseline_cfg import BaselineConfig


@CONFIGS.register_module(name="roi_relation_box_refine_edge_loss")
class RoIRelationBoxRefineEdgeLossConfig(BaselineConfig):
    TASK_NAME = "roi_relation_box_refine_edge_loss"

    MODEL = {
        **BaselineConfig.MODEL,
        "box_roi_relation_refinement": {
            "type": "RoIRelationBoxFeatureRefinement",
            "in_channels": 256,
            "relation_dim": 128,
            "geom_hidden_dim": 64,
            "top_k": 32,
            "residual_init": 0.1,
            "edge_loss_weight": 0.05,
            "max_edge_pairs": 4096,
        },
    }
