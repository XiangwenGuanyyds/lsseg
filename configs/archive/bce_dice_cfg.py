"""Mixed-scene + BCE/Dice mask loss.

Same as baseline except cfg.MODEL["mask_loss"] is set to
"bce_dice_loss". The mask target construction path stays identical to the
vanilla torchvision Mask R-CNN loss; only the final mask loss is split into
BCE and Dice terms, both returned through the MASK_LOSSES registry.

Run:
    python main.py --mode train --config bce_dice --exp_name <name>
    python main.py --mode test  --config bce_dice --exp_name <name>
"""

from utils.registry import CONFIGS

from .baseline_cfg import BaselineConfig


@CONFIGS.register_module(name="bce_dice")
class BCEDiceConfig(BaselineConfig):
    TASK_NAME = "bce_dice"

    MODEL = {
        **BaselineConfig.MODEL,
        "mask_loss": "bce_dice_loss",
    }

    # Explicitly keep the two restored mask terms at weight 1.0 so this config
    # remains self-documenting while preserving LossAggregator's default.
    LOSS = {
        "type": "LossAggregator",
        "loss_weights": {
            "loss_mask_bce": 1.0,
            "loss_mask_dice": 1.0,
        },
    }
