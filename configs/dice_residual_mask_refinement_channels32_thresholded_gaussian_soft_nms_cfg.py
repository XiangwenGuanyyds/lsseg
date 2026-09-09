"""Final-model inference with 32-channel refinement features."""

from utils.registry import CONFIGS

from .dice_w16_mask_refinement_feature_channels_32_cfg import (
    DiceW16MaskRefinementFeatureChannels32Config,
)


@CONFIGS.register_module(
    name=(
        "dice_residual_mask_refinement_channels32_"
        "thresholded_gaussian_soft_nms"
    )
)
class DiceResidualMaskRefinementChannels32ThresholdedGaussianSoftNMSConfig(
    DiceW16MaskRefinementFeatureChannels32Config
):
    MODEL = {
        **DiceW16MaskRefinementFeatureChannels32Config.MODEL,
        "box_nms": {
            "type": "thresholded_gaussian_soft_nms",
            "sigma": 0.5,
            "score_thresh_keep": 1e-3,
        },
    }
