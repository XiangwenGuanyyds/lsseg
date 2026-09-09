"""Baseline with BCE and boundary supervision at boundary weight 0.3."""

from utils.registry import CONFIGS

from .baseline_cfg import BaselineConfig


@CONFIGS.register_module(name="boundary_loss_w0_3")
class BoundaryLossW03Config(BaselineConfig):
    MODEL = {
        **BaselineConfig.MODEL,
        "mask_loss": "bce_boundary_loss",
    }
    LOSS = {
        "type": "LossAggregator",
        "loss_weights": {
            "loss_mask_bce": 1.0,
            "loss_mask_boundary": 0.3,
        },
    }
