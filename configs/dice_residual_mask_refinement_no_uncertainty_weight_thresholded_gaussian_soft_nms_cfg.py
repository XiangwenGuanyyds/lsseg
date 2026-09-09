"""Final-model inference without the uncertainty weight."""

from utils.registry import CONFIGS

from .dice_w16_mask_refinement_feature_channels_64_no_spatial_weight_cfg import (
    DiceW16MaskRefinementFeatureChannels64NoSpatialWeightConfig,
)


@CONFIGS.register_module(
    name=(
        "dice_residual_mask_refinement_no_uncertainty_weight_"
        "thresholded_gaussian_soft_nms"
    )
)
class DiceResidualMaskRefinementNoUncertaintyWeightThresholdedGaussianSoftNMSConfig(
    DiceW16MaskRefinementFeatureChannels64NoSpatialWeightConfig
):
    MODEL = {
        **DiceW16MaskRefinementFeatureChannels64NoSpatialWeightConfig.MODEL,
        "box_nms": {
            "type": "thresholded_gaussian_soft_nms",
            "sigma": 0.5,
            "score_thresh_keep": 1e-3,
        },
    }
