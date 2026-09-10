# Models and Thesis Modules

| Thesis component | Implementation |
| --- | --- |
| ResNet-50-FPN Mask R-CNN | `mask_rcnn.py`: `MaskRCNN` |
| RoI box prediction and post-processing | `heads/custom_roi_heads.py`: `CustomRoIHeads` |
| BCE mask loss and Dice loss | `losses/mask_rcnn_losses.py`: `maskrcnn_loss`, `dice_loss`, `bce_dice_loss_dict` |
| Weighted training objective | `losses/loss_aggregator.py`: `LossAggregator` |
| Residual Mask Refinement | `heads/residual_mask_refinement.py`: `ResidualMaskRefinementHead` |
| Standard NMS | `heads/box_nms.py`: `standard_nms` |
| Gaussian Soft-NMS | `heads/box_nms.py`: `gaussian_soft_nms` |
| Thresholded Gaussian Soft-NMS | `heads/box_nms.py`: `thresholded_gaussian_soft_nms` |

`MaskRCNN` constructs the backbone, RPN and prediction heads from torchvision's
`maskrcnn_resnet50_fpn`. `backbones/resnet.py` is a separate backbone wrapper.

The final refinement configuration uses `high_resolution_encoder_mode="conv1x1"`,
`refinement_feature_channels=64`, `use_uncertainty_weight=True`, and
`use_uncertainty_weight_in_predictor=False`. The weight scales residual logits
before they are added to the coarse mask logits.

Earlier experiments are retained in `attention/`, `native/`,
`heads/box_overlap_refinement.py` and `heads/roi_relation_refinement.py`.
Boundary mask loss and GIoU box loss are also retained as experimental options.

Saved experiment configs and checkpoints can contain earlier names. The CLI
loads saved settings through `utils/saved_config.py`; model-name conversion is
handled in `build.py` and `mask_rcnn.py`. Current code uses the names above;
saved results and experiment directory names are preserved.
