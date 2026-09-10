# Experiment Configurations

| Configuration | Stage | Experiment |
| --- | --- | --- |
| `baseline` | Training and testing | ResNet-50-FPN Mask R-CNN baseline with BCE mask loss and standard NMS |
| `dice` | Training and testing | Baseline + Dice loss, with standard NMS |
| `residual_mask_refinement` | Training and testing | Baseline + Residual Mask Refinement, with standard NMS |
| `dice_residual_mask_refinement` | Training and testing | Baseline + Dice loss + Residual Mask Refinement; final-model training, with standard NMS for validation and testing |
| `dice_residual_mask_refinement_without_uncertainty_weight` | Training and testing | Baseline + Dice loss + Residual Mask Refinement without uncertainty weighting, with standard NMS |
| `gaussian_soft_nms` | Testing only | Baseline + Gaussian Soft-NMS without an IoU threshold |
| `thresholded_gaussian_soft_nms` | Testing only | Baseline + Thresholded Gaussian Soft-NMS |
| `final_model` | Testing only | Final model: Baseline + Dice loss + Residual Mask Refinement + Thresholded Gaussian Soft-NMS |
| `final_model_without_refinement` | Testing only | Final-model ablation: Baseline + Dice loss + Thresholded Gaussian Soft-NMS, with refinement removed |
| `final_model_without_uncertainty_weight` | Testing only | Final-model ablation: Baseline + Dice loss + Residual Mask Refinement + Thresholded Gaussian Soft-NMS, with uncertainty weighting removed |
