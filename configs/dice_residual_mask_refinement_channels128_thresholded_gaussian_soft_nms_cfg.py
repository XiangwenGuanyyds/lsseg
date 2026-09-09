"""Final-model inference with 128-channel refinement features."""

from utils.registry import CONFIGS

from .dice_w16_mask_refinement_feature_channels_128_cfg import (
    DiceW16MaskRefinementFeatureChannels128Config,
)


@CONFIGS.register_module(
    name=(
        "dice_residual_mask_refinement_channels128_"
        "thresholded_gaussian_soft_nms"
    )
)
class DiceResidualMaskRefinementChannels128ThresholdedGaussianSoftNMSConfig(
    DiceW16MaskRefinementFeatureChannels128Config
):
    MODEL = {
        **DiceW16MaskRefinementFeatureChannels128Config.MODEL,
        "box_nms": {
            "type": "thresholded_gaussian_soft_nms",
            "sigma": 0.5,
            "score_thresh_keep": 1e-3,
        },
    }
