"""Residual Mask Refinement for Mask R-CNN.

P2 features and coarse mask logits go into the residual predictor.
The uncertainty weight scales the residual before adding it to the coarse mask.
Encoder size and weighting options come from config.
"""

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign

from utils import MODELS


@MODELS.register_module()
class ResidualMaskRefinementHead(nn.Module):
    """Refine one foreground mask-logit channel with high-resolution features."""

    def __init__(
        self,
        in_channels=256,
        hidden_channels=64,
        refinement_feature_channels=64,
        featmap_names=("0",),
        roi_output_size=28,
        sampling_ratio=2,
        output_size=56,
        high_resolution_encoder_mode="local",
        uncertainty_pool_kernel_size=3,
        foreground_class=1,
        detach_coarse_context=True,
        use_uncertainty_weight=True,
        use_uncertainty_weight_in_predictor=True,
    ):
        super().__init__()
        if (
            in_channels <= 0
            or hidden_channels <= 0
            or refinement_feature_channels <= 0
        ):
            raise ValueError("Feature dimensions must be positive.")
        if roi_output_size <= 0 or output_size <= 0:
            raise ValueError("RoI and output spatial sizes must be positive.")
        if output_size < roi_output_size:
            raise ValueError(
                "output_size must be greater than or equal to roi_output_size."
            )
        if uncertainty_pool_kernel_size <= 0 or uncertainty_pool_kernel_size % 2 == 0:
            raise ValueError("uncertainty_pool_kernel_size must be a positive odd integer.")
        if foreground_class < 0:
            raise ValueError("foreground_class must be non-negative.")
        if not featmap_names:
            raise ValueError("featmap_names must contain at least one FPN level.")
        valid_encoder_modes = {"local", "conv1x1", "identity"}
        if high_resolution_encoder_mode not in valid_encoder_modes:
            raise ValueError(
                "high_resolution_encoder_mode must be one of "
                f"{sorted(valid_encoder_modes)}, got "
                f"'{high_resolution_encoder_mode}'."
            )

        self.roi_output_size = (int(roi_output_size), int(roi_output_size))
        self.output_size = (int(output_size), int(output_size))
        self.uncertainty_pool_kernel_size = int(uncertainty_pool_kernel_size)
        self.foreground_class = int(foreground_class)
        self.detach_coarse_context = bool(detach_coarse_context)
        self.use_uncertainty_weight = bool(use_uncertainty_weight)
        # Predictor input uses the weight only when uncertainty weighting is enabled.
        self.use_uncertainty_weight_in_predictor = bool(
            self.use_uncertainty_weight and use_uncertainty_weight_in_predictor
        )
        self.featmap_names = tuple(str(name) for name in featmap_names)
        self.high_resolution_encoder_mode = str(high_resolution_encoder_mode)
        self.refinement_feature_channels = int(refinement_feature_channels)

        # Reuse torchvision's MultiScaleRoIAlign implementation unchanged.
        # Restricting featmap_names to ("0",) selects the existing FPN P2 map;
        # no torchvision pooling or coordinate-transform code is reimplemented.
        self.high_resolution_roi_pool = MultiScaleRoIAlign(
            featmap_names=list(self.featmap_names),
            output_size=self.roi_output_size,
            sampling_ratio=int(sampling_ratio),
        )

        if self.high_resolution_encoder_mode == "local":
            self.high_resolution_encoder = nn.Sequential(
                nn.Conv2d(int(in_channels), int(hidden_channels), kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    int(hidden_channels),
                    int(hidden_channels),
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    int(hidden_channels),
                    int(hidden_channels),
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(inplace=True),
            )
            residual_feature_channels = int(hidden_channels)
        elif self.high_resolution_encoder_mode == "conv1x1":
            # Combine feature channels with one 1x1 convolution.
            self.high_resolution_encoder = nn.Sequential(
                nn.Conv2d(
                    int(in_channels),
                    self.refinement_feature_channels,
                    kernel_size=1,
                ),
                nn.ReLU(inplace=True),
            )
            residual_feature_channels = self.refinement_feature_channels
        else:
            self.high_resolution_encoder = nn.Identity()
            residual_feature_channels = int(in_channels)

        residual_context_channels = 1 + int(
            self.use_uncertainty_weight_in_predictor
        )
        self.residual_predictor = nn.Sequential(
            nn.Conv2d(
                residual_feature_channels + residual_context_channels,
                int(hidden_channels),
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                int(hidden_channels),
                int(hidden_channels),
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(hidden_channels), 1, kernel_size=1),
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Zero initialization makes the optional path initially return the
        # coarse logits aligned to the configured refinement grid.
        nn.init.zeros_(self.residual_predictor[-1].weight)
        nn.init.zeros_(self.residual_predictor[-1].bias)

    def _align_to_output_size(self, tensor):
        """Resize a spatial tensor only when the configured grid requires it."""
        if tuple(tensor.shape[-2:]) == self.output_size:
            return tensor
        return F.interpolate(
            tensor,
            size=self.output_size,
            mode="bilinear",
            align_corners=False,
        )

    def _validate_inputs(self, features, boxes, image_shapes, coarse_logits):
        if not isinstance(features, Mapping):
            raise TypeError("features must be an FPN feature mapping.")
        missing = [name for name in self.featmap_names if name not in features]
        if missing:
            raise KeyError(f"Missing configured FPN feature maps: {missing}")
        if not isinstance(boxes, Sequence) or not isinstance(image_shapes, Sequence):
            raise TypeError("boxes and image_shapes must be per-image sequences.")
        if len(boxes) != len(image_shapes):
            raise ValueError("boxes and image_shapes must have the same length.")
        if coarse_logits.ndim != 4:
            raise ValueError(
                f"coarse_logits must have shape [N, K, H, W], got {tuple(coarse_logits.shape)}."
            )
        if any(
            size > target
            for size, target in zip(coarse_logits.shape[-2:], self.output_size)
        ):
            raise ValueError(
                "output_size must not be smaller than the coarse-mask spatial size."
            )
        roi_count = sum(int(boxes_per_image.shape[0]) for boxes_per_image in boxes)
        if roi_count != int(coarse_logits.shape[0]):
            raise ValueError(
                f"RoI count ({roi_count}) does not match coarse logits ({coarse_logits.shape[0]})."
            )
        if self.foreground_class >= int(coarse_logits.shape[1]):
            raise ValueError(
                f"foreground_class={self.foreground_class} is outside "
                f"the {coarse_logits.shape[1]} mask-logit channels."
            )

    def forward(self, features, boxes, image_shapes, coarse_logits):
        self._validate_inputs(features, boxes, image_shapes, coarse_logits)

        aligned_logits = self._align_to_output_size(coarse_logits)
        if coarse_logits.shape[0] == 0:
            return aligned_logits

        high_resolution_features = self.high_resolution_roi_pool(
            features,
            boxes,
            image_shapes,
        )
        high_resolution_features = self.high_resolution_encoder(
            high_resolution_features
        )
        high_resolution_features = self._align_to_output_size(
            high_resolution_features
        )

        coarse_instance_logits = aligned_logits[
            :, self.foreground_class : self.foreground_class + 1
        ]
        coarse_context = (
            coarse_instance_logits.detach()
            if self.detach_coarse_context
            else coarse_instance_logits
        )
        if self.use_uncertainty_weight:
            instance_mask_probability = coarse_context.sigmoid()
            uncertainty = (
                4.0 * instance_mask_probability * (1.0 - instance_mask_probability)
            )
            uncertainty_weight = F.max_pool2d(
                uncertainty,
                kernel_size=self.uncertainty_pool_kernel_size,
                stride=1,
                padding=self.uncertainty_pool_kernel_size // 2,
            )
        else:
            uncertainty_weight = None

        if self.use_uncertainty_weight_in_predictor:
            residual_input = torch.cat(
                (high_resolution_features, coarse_context, uncertainty_weight),
                dim=1,
            )
        else:
            residual_input = torch.cat(
                (high_resolution_features, coarse_context),
                dim=1,
            )

        residual_logits = self.residual_predictor(residual_input)
        if uncertainty_weight is None:
            refined_instance_logits = coarse_instance_logits + residual_logits
        else:
            refined_instance_logits = coarse_instance_logits + uncertainty_weight * residual_logits

        return torch.cat(
            (
                aligned_logits[:, : self.foreground_class],
                refined_instance_logits,
                aligned_logits[:, self.foreground_class + 1 :],
            ),
            dim=1,
        )
