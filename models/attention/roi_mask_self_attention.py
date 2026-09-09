"""RoI-level self-attention for the Mask R-CNN mask branch.

The module treats one complete RoI feature map as one token.  It is inserted
between ``mask_roi_pool`` and the original ``mask_head`` and returns the same
four-dimensional tensor shape as its input.
"""

import torch
import torch.nn as nn

from utils import MODELS


@MODELS.register_module()
class RoIMaskSelfAttention(nn.Module):
    """Model relations between RoIs from the same image.

    Each input RoI has shape ``[C, H, W]`` and is flattened into one vector.
    A learned projection maps that vector to one token of size ``token_dim``.
    Self-attention is applied independently to the RoI sequence of each image.
    The attended token is projected back to ``C`` channels, broadcast over the
    RoI spatial grid, and added to the original RoI feature map.
    """

    def __init__(
        self,
        in_channels=256,
        token_height=14,
        token_width=14,
        token_dim=256,
        num_heads=4,
        dropout=0.0,
        attention_scale=1.0,
    ):
        super().__init__()
        if token_height <= 0 or token_width <= 0:
            raise ValueError("token_height and token_width must be positive.")
        if token_dim <= 0:
            raise ValueError("token_dim must be positive.")
        if token_dim % num_heads != 0:
            raise ValueError(
                "RoIMaskSelfAttention requires token_dim to be divisible by "
                f"num_heads (got token_dim={token_dim}, num_heads={num_heads})."
            )
        if not isinstance(attention_scale, (int, float)):
            raise TypeError("attention_scale must be a fixed numeric value.")

        self.in_channels = int(in_channels)
        self.token_height = int(token_height)
        self.token_width = int(token_width)
        self.token_dim = int(token_dim)

        # One complete RoI feature map becomes one token.  With the default
        # mask RoI size, this is [256*14*14] -> [256].
        self.roi_to_token = nn.Linear(
            self.in_channels * self.token_height * self.token_width,
            self.token_dim,
        )
        self.norm = nn.LayerNorm(self.token_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.token_to_roi = nn.Linear(self.token_dim, self.in_channels)

        # This coefficient is intentionally fixed and configurable.  It is not
        # a trainable parameter; the experiment config controls its value.
        self.attention_scale = float(attention_scale)

    def forward(self, x, roi_counts=None):
        if x.numel() == 0:
            return x
        if x.dim() != 4:
            raise ValueError(
                "RoIMaskSelfAttention expects a 4D tensor "
                f"[num_rois, channels, height, width], got shape={tuple(x.shape)}."
            )

        num_rois, channels, height, width = x.shape
        expected = (self.in_channels, self.token_height, self.token_width)
        actual = (channels, height, width)
        if actual != expected:
            raise ValueError(
                "RoIMaskSelfAttention received an unexpected RoI feature shape: "
                f"expected [N, {expected[0]}, {expected[1]}, {expected[2]}], "
                f"got {tuple(x.shape)}."
            )

        if roi_counts is None:
            roi_counts = [num_rois]
        roi_counts = [int(count) for count in roi_counts]
        if any(count < 0 for count in roi_counts) or sum(roi_counts) != num_rois:
            raise ValueError(
                "roi_counts must be non-negative and sum to the number of RoIs "
                f"(got counts={roi_counts}, num_rois={num_rois})."
            )

        # Each RoI contributes one token.  RoIs from different images are
        # processed in separate sequences and therefore cannot attend to each
        # other.
        tokens = self.roi_to_token(x.flatten(1))  # [N, token_dim]

        outputs = []
        start = 0
        for count in roi_counts:
            if count == 0:
                continue

            sequence = tokens[start:start + count].unsqueeze(0)  # [1, R, D]
            sequence_norm = self.norm(sequence)
            attended, _ = self.attention(
                sequence_norm,
                sequence_norm,
                sequence_norm,
                need_weights=False,
            )
            attended = self.token_to_roi(attended.squeeze(0))  # [R, C]
            context = attended.unsqueeze(-1).unsqueeze(-1)  # [R, C, 1, 1]
            outputs.append(
                x[start:start + count] + self.attention_scale * context
            )
            start += count

        return torch.cat(outputs, dim=0) if outputs else x
