"""Dice and Residual Mask Refinement with fusion-only uncertainty weighting."""

from utils.registry import CONFIGS

from .dice_w16_mask_refinement_feature_channels_64_spatial_weight_fusion_only_cfg import (
    DiceW16MaskRefinementFeatureChannels64SpatialWeightFusionOnlyConfig,
)


@CONFIGS.register_module(
    name="dice_residual_mask_refinement_uncertainty_weight_fusion_only"
)
class DiceResidualMaskRefinementUncertaintyWeightFusionOnlyConfig(
    DiceW16MaskRefinementFeatureChannels64SpatialWeightFusionOnlyConfig
):
    pass
