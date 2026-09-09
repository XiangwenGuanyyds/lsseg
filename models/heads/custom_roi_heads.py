"""RoI heads, box experiments and NMS switch here.

Adapted from torchvision 0.22.1: detection.faster_rcnn and detection.roi_heads.
Mask loss code: models.losses.mask_rcnn_losses
"""

import torch
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import (
    RoIHeads,
    maskrcnn_inference,
)
from torchvision.ops import boxes as box_ops

# old scripts still import from here
from models.losses.mask_rcnn_losses import (
    bce_boundary_loss_dict,
    bce_dice_boundary_loss_dict,
    bce_dice_loss_dict,
    dice_loss_dict,
    maskrcnn_loss,
    maskrcnn_loss_dict,
    project_masks_on_boxes,
)


class CustomRoIHeads(RoIHeads):
    """RoIHeads with extra box options."""

    def box_head_forward(
        self,
        box_features,
        proposals=None,
        matched_idxs=None,
        labels=None,
    ):
        """RoI features in, class logits and box offsets out."""
        # print('[debug] roi size', box_features.shape)
        relation_refinement = getattr(
            self, "_box_roi_relation_refinement", None
        )
        self._box_roi_relation_loss = None
        if relation_refinement is not None:
            box_features = relation_refinement(
                box_features,
                proposals,
                matched_idxs=matched_idxs,
                labels=labels,
            )
            self._box_roi_relation_loss = getattr(
                relation_refinement, "last_edge_loss", None
            )

        # flatten first, features go through 2 FC
        box_features = box_features.flatten(start_dim=1)
        box_features = F.relu(self.box_head.fc6(box_features))
        box_features = F.relu(self.box_head.fc7(box_features))

        # from FastRCNNPredictor
        if box_features.dim() == 4:
            box_features = box_features.flatten(start_dim=1)
        class_logits = self.box_predictor.cls_score(box_features)

        regression_features = box_features
        overlap_refinement = getattr(self, "_box_overlap_refinement", None)
        if overlap_refinement is not None:
            regression_features = overlap_refinement(
                regression_features, proposals
            )
        box_regression = self.box_predictor.bbox_pred(regression_features)
        # print("[debug] cls / box:", class_logits.shape, box_regression.shape)
        return class_logits, box_regression

    def postprocess_detections(
        self,
        class_logits,
        box_regression,
        proposals,
        image_shapes,
    ):
        """Box decode, filtering, then NMS."""
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        self._last_pre_nms_candidates = []
        self._last_post_nms_candidates = []

        boxes_per_image = [boxes.shape[0] for boxes in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, dim=-1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, dim=0)
        pred_scores_list = pred_scores.split(boxes_per_image, dim=0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(
            pred_boxes_list, pred_scores_list, image_shapes
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # class 0 is background, skip
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            candidate_ids = torch.arange(boxes.shape[0], device=device)

            keep = torch.where(scores > self.score_thresh)[0]
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            candidate_ids = candidate_ids[keep]

            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            candidate_ids = candidate_ids[keep]

            # boxes before nms, use for recall later
            self._last_pre_nms_candidates.append(
                {
                    "boxes": boxes.detach(),
                    "scores": scores.detach(),
                    "labels": labels.detach(),
                    "candidate_ids": candidate_ids.detach(),
                }
            )

            keep, kept_scores = self._nms_fn(
                boxes,
                scores,
                labels,
                nms_thresh=self.nms_thresh,
                **self._nms_kwargs,
            )
            keep = keep[: self.detections_per_img]
            boxes = boxes[keep]
            labels = labels[keep]
            scores = kept_scores[: self.detections_per_img]
            candidate_ids = candidate_ids[keep]
            # print('[debug] nms done', len(boxes), scores[:5])

            self._last_post_nms_candidates.append(
                {
                    "boxes": boxes.detach(),
                    "scores": scores.detach(),
                    "labels": labels.detach(),
                    "candidate_ids": candidate_ids.detach(),
                }
            )

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels


def fastrcnn_loss(
    class_logits,
    box_regression,
    labels,
    regression_targets,
    proposals,
    box_coder,
    reg_loss_fn,
):
    """cls loss + box loss, reg_loss_fn comes from config."""
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = F.cross_entropy(class_logits, labels)

    positive_indices = torch.where(labels > 0)[0]
    # print("[debug] pos count", positive_indices.numel(), "all", labels.numel())
    positive_labels = labels[positive_indices]
    num_samples = class_logits.shape[0]
    box_regression = box_regression.reshape(
        num_samples, box_regression.size(-1) // 4, 4
    )

    proposals = torch.cat(proposals, dim=0)
    box_loss = reg_loss_fn(
        box_regression[positive_indices, positive_labels],
        regression_targets[positive_indices],
        proposals[positive_indices],
        box_coder,
    )
    box_loss = box_loss / labels.numel()  # divide by all samples here
    return classification_loss, box_loss


__all__ = [
    "CustomRoIHeads",
    "bce_boundary_loss_dict",
    "bce_dice_boundary_loss_dict",
    "bce_dice_loss_dict",
    "dice_loss_dict",
    "fastrcnn_loss",
    "maskrcnn_inference",
    "maskrcnn_loss",
    "maskrcnn_loss_dict",
    "project_masks_on_boxes",
]
