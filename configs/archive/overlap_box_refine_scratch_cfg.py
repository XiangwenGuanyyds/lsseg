"""Mixed-scene overlap-aware box refinement without backbone pretraining.

This config is only for checking whether the structural change behaves
differently when the whole Mask R-CNN model is trained from random
initialization. Existing configs keep the original backbone pretraining path.

Run:
    python main.py --mode train --config overlap_box_refine_scratch --exp_name <name>
    python main.py --mode test  --config overlap_box_refine_scratch --exp_name <name>
"""

from utils.registry import CONFIGS

from .overlap_box_refine_cfg import OverlapBoxRefineConfig


@CONFIGS.register_module(name="overlap_box_refine_scratch")
class OverlapBoxRefineScratchConfig(OverlapBoxRefineConfig):
    TASK_NAME = "overlap_box_refine_scratch"

    MODEL = {
        **OverlapBoxRefineConfig.MODEL,
        "resnet50_body_weights": None,
        "trainable_resnet_layers": 5,
    }
