# Torchvision-derived portions: BSD-3-Clause.
# Copyright (c) Soumith Chintala 2016,
# All rights reserved.
# Full licence: LICENSES/torchvision.txt (repository root).

"""ResNet-50-FPN Mask R-CNN used by the baseline and final model.

Forward pipeline adapted from torchvision v0.22.1:
https://github.com/pytorch/vision/blob/v0.22.1/torchvision/models/detection/generalized_rcnn.py
    GeneralizedRCNN.forward.
https://github.com/pytorch/vision/blob/v0.22.1/torchvision/models/detection/roi_heads.py
    RoIHeads.forward: box and mask branches.

Local changes: configurable losses and NMS, Residual Mask Refinement,
optional attention and box refinement, and evaluation diagnostics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.roi_heads import maskrcnn_inference

from evaluation.pipeline_analysis import (
    accumulate_postprocessing_recall,
    accumulate_rpn_coverage,
)
from models.heads.custom_roi_heads import CustomRoIHeads, fastrcnn_loss
from utils import BOX_NMS, BOX_REG_LOSSES, MASK_LOSSES, MODELS


@MODELS.register_module()
class MaskRCNN(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        num_classes = cfg.MODEL.get("num_classes",
                                     cfg.MODEL.get("head", {}).get("num_classes"))
        if num_classes is None:
            raise ValueError(
                "cfg.MODEL must specify 'num_classes' "
                "(or 'head.num_classes' for legacy configs)."
            )

        # ResNet weights and trainable stages.
        full_maskrcnn_weights = cfg.MODEL.get("full_maskrcnn_weights", None)
        resnet50_body_weights = cfg.MODEL.get(
            "resnet50_body_weights", "IMAGENET1K_V2"
        )
        trainable_resnet_layers = cfg.MODEL.get("trainable_resnet_layers", 3)

        full_maskrcnn_weights = self._normalize_weight_setting(
            full_maskrcnn_weights
        )
        resnet50_body_weights = self._normalize_weight_setting(
            resnet50_body_weights
        )
        if full_maskrcnn_weights is not None and resnet50_body_weights is not None:
            raise ValueError(
                "Choose either complete Mask R-CNN weights or ResNet-50 body "
                "weights, not both."
            )
        if not 0 <= trainable_resnet_layers <= 5:
            raise ValueError("trainable_resnet_layers must be between 0 and 5")
        if (
            full_maskrcnn_weights is None
            and resnet50_body_weights is None
            and trainable_resnet_layers != 5
        ):
            raise ValueError(
                "A randomly initialized ResNet-50 must train all 5 layers; "
                "set trainable_resnet_layers=5."
            )

        self._print_initialization_summary(
            full_maskrcnn_weights,
            resnet50_body_weights,
            trainable_resnet_layers,
        )

        # Random initialisation: train all ResNet stages.
        torchvision_trainable_layers = trainable_resnet_layers
        if full_maskrcnn_weights is None and resnet50_body_weights is None:
            torchvision_trainable_layers = None

        m = maskrcnn_resnet50_fpn(
            # Full detector weights.
            weights=full_maskrcnn_weights,
            # ResNet body weights.
            weights_backbone=resnet50_body_weights,
            trainable_backbone_layers=torchvision_trainable_layers,
            num_classes=num_classes,
        )

        # Image transform, backbone, RPN.
        self.transform = m.transform
        self.backbone  = m.backbone
        self.rpn       = m.rpn

        # Configurable box loss and NMS.
        self.roi_heads = m.roi_heads
        self.roi_heads.__class__ = CustomRoIHeads

        # Box head called through roi_heads.box_head_forward.
        self.box_roi_pool = m.roi_heads.box_roi_pool

        # Mask branch.
        self.mask_roi_pool  = m.roi_heads.mask_roi_pool
        self.mask_head      = m.roi_heads.mask_head
        self.mask_predictor = m.roi_heads.mask_predictor

        # FPN attention, before RPN and RoI heads.
        self.attention = self._build_attention(cfg)

        # RoI attention, between mask pooling and mask head.
        self.mask_attention = self._build_mask_attention(cfg)
        self.mask_correction = self._build_mask_correction(cfg)
        self.residual_mask_refinement = self._build_residual_mask_refinement(cfg)

        # Proposal overlap features for box regression.
        self.roi_heads._box_overlap_refinement = self._build_box_overlap_refinement(cfg)

        # RoI relations in the box branch.
        self.roi_heads._box_roi_relation_refinement = self._build_box_roi_relation_refinement(cfg)

        # Epoch, output paths and diagnostic flags from the runner.
        self.state = None

        self.debug = getattr(cfg, "DEBUG", {}) or {}

        # Mask loss from config; BCE by default.
        mask_loss_name = cfg.MODEL.get("mask_loss", "maskrcnn_loss")
        self.mask_loss_fn = MASK_LOSSES.get(mask_loss_name)
        if self.mask_loss_fn is None:
            raise KeyError(
                f"cfg.MODEL['mask_loss']='{mask_loss_name}' not in MASK_LOSSES. "
                f"available: {MASK_LOSSES.keys()}"
            )

        # NMS selection.
        nms_cfg = cfg.MODEL.get("box_nms", {"type": "standard_nms"})
        nms_kwargs = dict(nms_cfg)
        nms_type = nms_kwargs.pop("type")
        nms_thresh = nms_kwargs.pop("nms_thresh", None)
        if nms_thresh is not None:
            nms_thresh = float(nms_thresh)
            if not 0.0 <= nms_thresh <= 1.0:
                raise ValueError(
                    "cfg.MODEL['box_nms'].nms_thresh must be in [0, 1]"
                )
            self.roi_heads.nms_thresh = nms_thresh
        nms_fn = BOX_NMS.get(nms_type)
        if nms_fn is None:
            raise KeyError(
                f"cfg.MODEL['box_nms'].type='{nms_type}' not in BOX_NMS. "
                f"available: {BOX_NMS.keys()}"
            )
        self.roi_heads._nms_fn     = nms_fn
        self.roi_heads._nms_kwargs = nms_kwargs

        # Box regression loss, default Smooth L1.
        reg_loss_name = cfg.MODEL.get("box_reg_loss", "smooth_l1")
        self.box_reg_loss_fn = BOX_REG_LOSSES.get(reg_loss_name)
        if self.box_reg_loss_fn is None:
            raise KeyError(
                f"cfg.MODEL['box_reg_loss']='{reg_loss_name}' not in BOX_REG_LOSSES. "
                f"available: {BOX_REG_LOSSES.keys()}"
            )

    @staticmethod
    def _normalize_weight_setting(value):
        if isinstance(value, str) and value.lower() in {"none", "null"}:
            return None
        return value

    @staticmethod
    def _print_initialization_summary(
        full_maskrcnn_weights,
        resnet50_body_weights,
        trainable_resnet_layers,
    ):
        print("[model initialization before optional checkpoint loading]")
        print(f"  complete Mask R-CNN weights : {full_maskrcnn_weights or 'None'}")
        if full_maskrcnn_weights is None:
            print(f"  ResNet-50 body weights      : {resnet50_body_weights or 'None'}")
            print("  FPN weights                 : random initialization")
            print("  RPN weights                 : random initialization")
            print("  RoI box/mask head weights   : random initialization")
        else:
            print("  ResNet/FPN/RPN/RoI heads    : loaded from complete weights")
        ordered_layers = ["layer4", "layer3", "layer2", "layer1", "conv1/bn1"]
        trainable = list(reversed(ordered_layers[:trainable_resnet_layers]))
        frozen = list(reversed(ordered_layers[trainable_resnet_layers:]))
        print(f"  trainable ResNet layers     : {', '.join(trainable) or 'None'}")
        print(f"  frozen ResNet layers        : {', '.join(frozen) or 'None'}")

    def _build_attention(self, cfg):
        attn_cfg = cfg.MODEL.get("attention", None)
        if not attn_cfg:
            return nn.Identity()

        attn_kwargs = dict(attn_cfg)
        attn_type = attn_kwargs.pop("type")
        attn_cls = MODELS.get(attn_type)
        if attn_cls is None:
            raise KeyError(
                f"cfg.MODEL['attention'].type='{attn_type}' not in MODELS. "
                f"available: {MODELS.keys()}"
            )
        return attn_cls(**attn_kwargs)

    def _build_mask_attention(self, cfg):
        attn_cfg = cfg.MODEL.get("mask_attention", None)
        if not attn_cfg:
            return nn.Identity()

        attn_kwargs = dict(attn_cfg)
        attn_type = attn_kwargs.pop("type")
        attn_cls = MODELS.get(attn_type)
        if attn_cls is None:
            raise KeyError(
                f"cfg.MODEL['mask_attention'].type='{attn_type}' not in MODELS. "
                f"available: {MODELS.keys()}"
            )
        return attn_cls(**attn_kwargs)

    def _build_mask_correction(self, cfg):
        correction_cfg = cfg.MODEL.get("mask_correction", None)
        if not correction_cfg:
            return None

        correction_kwargs = dict(correction_cfg)
        correction_type = correction_kwargs.pop("type")
        correction_cls = MODELS.get(correction_type)
        if correction_cls is None:
            raise KeyError(
                f"cfg.MODEL['mask_correction'].type='{correction_type}' "
                f"not in MODELS. available: {MODELS.keys()}"
            )
        return correction_cls(**correction_kwargs)

    def _build_residual_mask_refinement(self, cfg):
        refinement_cfg = cfg.MODEL.get("residual_mask_refinement")
        if not refinement_cfg:
            return None

        refinement_kwargs = dict(refinement_cfg)
        refinement_type = refinement_kwargs.pop("type")
        refinement_cls = MODELS.get(refinement_type)
        if refinement_cls is None:
            raise KeyError(
                "cfg.MODEL['residual_mask_refinement'].type="
                f"'{refinement_type}' not in MODELS. available: {MODELS.keys()}"
            )
        return refinement_cls(**refinement_kwargs)

    def _build_box_overlap_refinement(self, cfg):
        refine_cfg = cfg.MODEL.get("box_overlap_refinement", None)
        if not refine_cfg:
            return None

        refine_kwargs = dict(refine_cfg)
        refine_type = refine_kwargs.pop("type")
        refine_cls = MODELS.get(refine_type)
        if refine_cls is None:
            raise KeyError(
                f"cfg.MODEL['box_overlap_refinement'].type='{refine_type}' not in MODELS. "
                f"available: {MODELS.keys()}"
            )
        return refine_cls(**refine_kwargs)

    def _build_box_roi_relation_refinement(self, cfg):
        refine_cfg = cfg.MODEL.get("box_roi_relation_refinement", None)
        if not refine_cfg:
            return None

        refine_kwargs = dict(refine_cfg)
        refine_type = refine_kwargs.pop("type")
        refine_cls = MODELS.get(refine_type)
        if refine_cls is None:
            raise KeyError(
                f"cfg.MODEL['box_roi_relation_refinement'].type='{refine_type}' not in MODELS. "
                f"available: {MODELS.keys()}"
            )
        return refine_cls(**refine_kwargs)

    def forward(self, x, targets=None):
        original_image_sizes = [(img.shape[-2], img.shape[-1]) for img in x]
        diagnostics = self.state.get("diagnostics", {}) if self.state else {}

        # Transform and FPN features.
        images, targets = self.transform(x, targets)
        # print('[debug] resized', images.tensors.shape, images.image_sizes)
        features = self.backbone(images.tensors)
        # print("[debug] FPN", {k: v.shape for k, v in features.items()})
        features = self.attention(features)
        proposals, proposal_losses = self.rpn(images, features, targets)
        accumulate_rpn_coverage(proposals, targets, self.state)

        # Training RoI samples.
        if self.training:
            proposals, matched_idxs, labels, regression_targets = \
                self.roi_heads.select_training_samples(proposals, targets)
        else:
            matched_idxs = labels = regression_targets = None

        # Box branch
        box_features = self.box_roi_pool(features, proposals, images.image_sizes)
        # print('[debug] box RoIs:', box_features.shape)
        class_logits, box_regression = self.roi_heads.box_head_forward(
            box_features,
            proposals,
            matched_idxs=matched_idxs,
            labels=labels,
        )
        # print("[debug]", class_logits.shape, "box offsets", box_regression.shape)

        detector_losses = {}
        result = []
        if self.training:
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets,
                proposals, self.roi_heads.box_coder, self.box_reg_loss_fn,
            )
            detector_losses["loss_classifier"] = loss_classifier
            detector_losses["loss_box_reg"]    = loss_box_reg
            relation_loss = getattr(self.roi_heads, "_box_roi_relation_loss", None)
            if relation_loss is not None:
                detector_losses["loss_roi_relation"] = relation_loss
        else:
            boxes, scores, det_labels = self.roi_heads.postprocess_detections(
                class_logits, box_regression, proposals, images.image_sizes,
            )
            for i in range(len(boxes)):
                result.append({
                    "boxes":  boxes[i],
                    "labels": det_labels[i],
                    "scores": scores[i],
                })
            if diagnostics.get("postprocessing_recall", False):
                accumulate_postprocessing_recall(
                    self.roi_heads._last_pre_nms_candidates,
                    self.roi_heads._last_post_nms_candidates,
                    targets,
                    self.state,
                )
        # Mask branch.
        if self.training:
            mask_proposals = []
            pos_matched_idxs = []
            for img_id in range(len(proposals)):
                pos = torch.where(labels[img_id] > 0)[0]
                mask_proposals.append(proposals[img_id][pos])
                pos_matched_idxs.append(matched_idxs[img_id][pos])
        else:
            mask_proposals = [r["boxes"] for r in result]
            pos_matched_idxs = None

        # print('[debug] mask counts', [len(p) for p in mask_proposals])
        mask_features = self.mask_roi_pool(features, mask_proposals, images.image_sizes)
        # print("[debug] pooled masks", mask_features.shape)
        if isinstance(self.mask_attention, nn.Identity):
            mask_features = self.mask_attention(mask_features)
        else:
            roi_counts = [int(p.shape[0]) for p in mask_proposals]
            mask_features = self.mask_attention(mask_features, roi_counts=roi_counts)
        mask_delta = None
        if self.mask_correction is not None:
            roi_counts = [int(p.shape[0]) for p in mask_proposals]
            mask_delta = self.mask_correction(
                mask_features,
                roi_counts=roi_counts,
                roi_boxes=mask_proposals,
            )
        mask_features = self.mask_head(mask_features)
        mask_logits   = self.mask_predictor(mask_features)
        # [num_rois, num_classes, 28, 28]
        # print('[debug] mask logits', mask_logits.shape)
        if mask_delta is not None and mask_delta.numel() > 0:
            mask_delta = F.interpolate(
                mask_delta,
                size=mask_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            mask_logits = mask_logits.clone()
            mask_logits[:, self.mask_correction.foreground_class : self.mask_correction.foreground_class + 1] += mask_delta
        if self.residual_mask_refinement is not None:
            mask_logits = self.residual_mask_refinement(
                features,
                mask_proposals,
                images.image_sizes,
                mask_logits,
            )

        if self.training:
            gt_masks  = [t["masks"]  for t in targets]
            gt_labels = [t["labels"] for t in targets]
            mask_loss_dict = self.mask_loss_fn(
                mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs,
            )
            detector_losses.update(mask_loss_dict)
        else:
            det_labels_list = [r["labels"] for r in result]
            masks_probs = maskrcnn_inference(mask_logits, det_labels_list)
            for mask_prob, r in zip(masks_probs, result):
                r["masks"] = mask_prob

        # Back to original image size.
        if not self.training:
            result = self.transform.postprocess(
                result, images.image_sizes, original_image_sizes,
            )

        if self.training:
            return {**proposal_losses, **detector_losses}
        return result
