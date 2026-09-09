"""Mask R-CNN mask targets and configurable mask losses.

The target projection follows torchvision's Mask R-CNN implementation. All
mask-loss variants used by experiment configurations are registered here so
there is one implementation path for BCE, Dice, and boundary supervision.
"""

from scipy.ndimage import distance_transform_edt
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

from utils import MASK_LOSSES


def project_masks_on_boxes(gt_masks, boxes, matched_idxs, mask_size):
    """Crop and resize matched ground-truth masks to the mask output size.

    This function is copied from torchvision 0.22.1
    ``torchvision.models.detection.roi_heads.project_masks_on_boxes``.
    Only argument names and this documentation were clarified.
    """
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    gt_masks = gt_masks[:, None].to(rois)
    return roi_align(
        gt_masks,
        rois,
        (mask_size, mask_size),
        spatial_scale=1.0,
    )[:, 0]


def _prepare_mask_loss_inputs(
    mask_logits,
    proposals,
    gt_masks,
    gt_labels,
    mask_matched_idxs,
):
    """Select the matched-class logits and construct their mask targets."""
    mask_size = mask_logits.shape[-1]
    labels = [
        labels_per_image[matched_per_image]
        for labels_per_image, matched_per_image in zip(
            gt_labels, mask_matched_idxs
        )
    ]
    mask_targets = [
        project_masks_on_boxes(
            masks_per_image,
            proposals_per_image,
            matched_per_image,
            mask_size,
        )
        for masks_per_image, proposals_per_image, matched_per_image in zip(
            gt_masks, proposals, mask_matched_idxs
        )
    ]

    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)
    if mask_targets.numel() == 0:
        return None, mask_targets

    row_indices = torch.arange(labels.shape[0], device=labels.device)
    pred_mask_logits = mask_logits[row_indices, labels]
    return pred_mask_logits, mask_targets


def maskrcnn_loss(
    mask_logits,
    proposals,
    gt_masks,
    gt_labels,
    mask_matched_idxs,
):
    """Compute the standard Mask R-CNN binary cross-entropy mask loss."""
    pred_mask_logits, mask_targets = _prepare_mask_loss_inputs(
        mask_logits,
        proposals,
        gt_masks,
        gt_labels,
        mask_matched_idxs,
    )
    if pred_mask_logits is None:
        return mask_logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(pred_mask_logits, mask_targets)


def dice_loss(pred_mask_logits, mask_targets, smooth=1.0):
    """Compute per-instance Dice loss and average over positive RoIs."""
    if pred_mask_logits.numel() == 0:
        return pred_mask_logits.sum() * 0.0

    pred_probs = torch.sigmoid(pred_mask_logits).flatten(1)
    targets = mask_targets.flatten(1)
    intersection = (pred_probs * targets).sum(dim=1)
    denominator = pred_probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return (1.0 - dice).mean()


def _compute_distance_maps(mask_targets):
    """Compute foreground and background distances to each target boundary."""
    binary_masks = (mask_targets > 0.5).detach().cpu().numpy()
    dist_in = torch.zeros_like(mask_targets)
    dist_out = torch.zeros_like(mask_targets)

    for index, mask in enumerate(binary_masks):
        if mask.sum() == 0 or mask.sum() == mask.size:
            continue
        dist_in[index] = torch.from_numpy(
            distance_transform_edt(mask)
        ).to(mask_targets)
        dist_out[index] = torch.from_numpy(
            distance_transform_edt(~mask)
        ).to(mask_targets)

    return dist_in, dist_out


def boundary_loss(pred_mask_logits, mask_targets):
    """Compute the non-negative distance-transform boundary loss.

    The formulation is equivalent up to a parameter-independent constant to
    the signed boundary loss described by Kervadec et al. (MIDL 2019).
    """
    if pred_mask_logits.numel() == 0:
        return pred_mask_logits.sum() * 0.0

    dist_in, dist_out = _compute_distance_maps(mask_targets)
    pred_probs = torch.sigmoid(pred_mask_logits)
    return (
        (1.0 - pred_probs) * dist_in + pred_probs * dist_out
    ).mean()


def _mask_terms(
    mask_logits,
    proposals,
    gt_masks,
    gt_labels,
    mask_matched_idxs,
):
    return _prepare_mask_loss_inputs(
        mask_logits,
        proposals,
        gt_masks,
        gt_labels,
        mask_matched_idxs,
    )


@MASK_LOSSES.register_module(name="maskrcnn_loss")
def maskrcnn_loss_dict(
    mask_logits,
    proposals,
    gt_masks,
    gt_labels,
    mask_matched_idxs,
):
    """Return the standard BCE mask loss using the registry contract."""
    return {
        "loss_mask": maskrcnn_loss(
            mask_logits,
            proposals,
            gt_masks,
            gt_labels,
            mask_matched_idxs,
        )
    }


@MASK_LOSSES.register_module(name="bce_dice_loss")
def bce_dice_loss_dict(
    mask_logits,
    proposals,
    gt_masks,
    gt_labels,
    mask_matched_idxs,
):
    """Return BCE and per-instance Dice mask-loss terms."""
    pred_logits, targets = _mask_terms(
        mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs
    )
    if pred_logits is None:
        zero = mask_logits.sum() * 0.0
        return {"loss_mask_bce": zero, "loss_mask_dice": zero}
    return {
        "loss_mask_bce": F.binary_cross_entropy_with_logits(
            pred_logits, targets
        ),
        "loss_mask_dice": dice_loss(pred_logits, targets),
    }


@MASK_LOSSES.register_module(name="dice_loss")
def dice_loss_dict(
    mask_logits,
    proposals,
    gt_masks,
    gt_labels,
    mask_matched_idxs,
):
    """Return per-instance Dice mask loss without BCE."""
    pred_logits, targets = _mask_terms(
        mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs
    )
    if pred_logits is None:
        return {"loss_mask_dice": mask_logits.sum() * 0.0}
    return {"loss_mask_dice": dice_loss(pred_logits, targets)}


@MASK_LOSSES.register_module(name="bce_boundary_loss")
def bce_boundary_loss_dict(
    mask_logits,
    proposals,
    gt_masks,
    gt_labels,
    mask_matched_idxs,
):
    """Return BCE and distance-transform boundary mask-loss terms."""
    pred_logits, targets = _mask_terms(
        mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs
    )
    if pred_logits is None:
        zero = mask_logits.sum() * 0.0
        return {"loss_mask_bce": zero, "loss_mask_boundary": zero}
    return {
        "loss_mask_bce": F.binary_cross_entropy_with_logits(
            pred_logits, targets
        ),
        "loss_mask_boundary": boundary_loss(pred_logits, targets),
    }


@MASK_LOSSES.register_module(name="bce_dice_boundary_loss")
def bce_dice_boundary_loss_dict(
    mask_logits,
    proposals,
    gt_masks,
    gt_labels,
    mask_matched_idxs,
):
    """Return BCE, Dice, and boundary mask-loss terms."""
    pred_logits, targets = _mask_terms(
        mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs
    )
    if pred_logits is None:
        zero = mask_logits.sum() * 0.0
        return {
            "loss_mask_bce": zero,
            "loss_mask_dice": zero,
            "loss_mask_boundary": zero,
        }
    return {
        "loss_mask_bce": F.binary_cross_entropy_with_logits(
            pred_logits, targets
        ),
        "loss_mask_dice": dice_loss(pred_logits, targets),
        "loss_mask_boundary": boundary_loss(pred_logits, targets),
    }
