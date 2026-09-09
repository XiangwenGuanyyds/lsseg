# Experiment configurations

Some available configuration names:

- `baseline`
- `gated_residual_refinement`
- `dice_w16`
- `gaussian_soft_nms`
- `thresholded_gaussian_soft_nms`
- `dice_w16_gaussian_soft_nms`
- `dice_w16_thresholded_gaussian_soft_nms`
- `dice_w16_gated_residual_refinement`
- `boundary_loss_w0_3`
- `cbam`

Residual Mask Refinement is configured through `MODEL["residual_mask_refinement"]`
with `type: "ResidualMaskRefinementHead"`. The implementation is in
`models/heads/residual_mask_refinement.py`.
Existing experiment names are kept. Older configs and checkpoints using
`gated_residual_refinement` can still be loaded.

Training example:

```bash
./lsseg train dice_w16_seed0 --config dice_w16 --seed 0
```

Test-time post-processing example:

```bash
./lsseg test baseline_seed0 --config thresholded_gaussian_soft_nms \
  --result-name thresholded_gaussian_soft_nms_seed0
```

Earlier exploratory configurations are stored in `configs/archive/`. They are
not registered and are not available through `--config`.
