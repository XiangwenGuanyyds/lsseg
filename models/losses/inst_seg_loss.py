import torch
import torch.nn as nn

from utils import LOSSES


@LOSSES.register_module()
class LossAggregator(nn.Module):
    """Apply configured weights and sum the model's training losses."""

    def __init__(self, loss_weights=None):
        super().__init__()
        self.loss_weights = dict(loss_weights or {})

    def forward(self, loss_dict):
        """Return the weighted total and detached scalar values for logging."""
        total_loss = 0
        weighted_losses = {}

        for name, value in loss_dict.items():
            weight = self.loss_weights.get(name, 1.0)
            weighted_value = value * weight
            total_loss += weighted_value
            weighted_losses[name] = weighted_value.item()

        return total_loss, weighted_losses
