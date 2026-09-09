"""Box regression losses, selected by MODEL["box_reg_loss"].

Smooth L1 is used in the baseline and final model. GIoU is an earlier experiment.

Inputs contain positive RoIs:
    pred_deltas [N_pos, 4]: predictions selected by ground-truth class
    target_deltas [N_pos, 4]: encoded ground-truth targets
    proposals [N_pos, 4]: proposal boxes in xyxy format
    box_coder: torchvision BoxCoder for decoding deltas

Each function returns a scalar loss sum. custom_roi_heads.fastrcnn_loss
divides it by the total number of sampled RoIs.
"""

import torch
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou

from utils import BOX_REG_LOSSES


@BOX_REG_LOSSES.register_module(name="smooth_l1")
def smooth_l1_box_reg(pred_deltas, target_deltas, proposals, box_coder):
    """Standard Smooth L1, beta=1/9, summed over positive RoIs."""
    # print("[debug] deltas", pred_deltas.shape, target_deltas.shape)
    # [debug] deltas torch.Size([2, 4]) torch.Size([2, 4])
    return F.smooth_l1_loss(pred_deltas, target_deltas, beta=1 / 9, reduction="sum")


@BOX_REG_LOSSES.register_module(name="giou")
def giou_box_reg(pred_deltas, target_deltas, proposals, box_coder):
    """Sum of 1 - GIoU for matched boxes (Rezatofighi et al., 2019).

    Tests overlap-based supervision for box localisation.
    """
    # decode to box coordinates first
    pred_boxes = box_coder.decode_single(pred_deltas, proposals)
    gt_boxes   = box_coder.decode_single(target_deltas, proposals)
    # diagonal gives the matched pairs
    giou = generalized_box_iou(pred_boxes, gt_boxes).diagonal()
    # print("[debug] giou", giou.detach())
    # [debug] giou tensor([0.8182, 1.0000])
    return (1 - giou).sum()
