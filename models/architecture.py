"""Mask R-CNN decomposed into its top-level torchvision sub-modules.

Builds a stock ``maskrcnn_resnet50_fpn`` and exposes ALL its replaceable
pieces at the architecture level, so each can be subclassed, hooked, or
swapped independently:

    self.transform          GeneralizedRCNNTransform
    self.backbone           BackboneWithFPN
    self.rpn                RegionProposalNetwork
    self.roi_heads          CustomRoIHeads  (subclass of torchvision RoIHeads;
                                             default body == vanilla; future
                                             DH variants override it)
    self.box_roi_pool       MultiScaleRoIAlign      ─ box branch entry
    self.mask_roi_pool      MultiScaleRoIAlign      ┐
    self.mask_head          MaskRCNNHeads           │ mask branch
    self.mask_predictor     MaskRCNNPredictor       ┘

Box branch internals (box_head + box_predictor module calls) are wrapped
inside ``self.roi_heads.box_head_forward`` so future variants override
that one method instead of touching ``Architecture.forward``.

Forward mirrors ``GeneralizedRCNN.forward`` for transform / backbone /
rpn, then expands ``RoIHeads.forward`` inline so the box and mask flows
are visible side-by-side at this level.
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
class Architecture(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        num_classes = cfg.MODEL.get("num_classes",
                                     cfg.MODEL.get("head", {}).get("num_classes"))
        if num_classes is None:
            raise ValueError(
                "cfg.MODEL must specify 'num_classes' "
                "(or 'head.num_classes' for legacy configs)."
            )

        # Initialization is named by the module that actually receives the
        # weights. In torchvision, ``weights_backbone`` initializes only the
        # ResNet body; the FPN is constructed afterwards and starts randomly.
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

        # torchvision always trains all five ResNet stages when neither set
        # of weights is loaded. Passing None selects that behavior without its
        # otherwise unavoidable warning; the validation above keeps it explicit.
        torchvision_trainable_layers = trainable_resnet_layers
        if full_maskrcnn_weights is None and resnet50_body_weights is None:
            torchvision_trainable_layers = None

        m = maskrcnn_resnet50_fpn(
            # ``weights`` loads a complete detection checkpoint, including
            # ResNet, FPN, RPN, box head, and mask head.
            weights=full_maskrcnn_weights,
            # Despite torchvision's name, this argument initializes only the
            # ResNet-50 body. It does not initialize the FPN.
            weights_backbone=resnet50_body_weights,
            trainable_backbone_layers=torchvision_trainable_layers,
            num_classes=num_classes,
        )

        # Top-level pre-RoI sub-modules.
        self.transform = m.transform
        self.backbone  = m.backbone
        self.rpn       = m.rpn

        # RoIHeads is always promoted to CustomRoIHeads — its body is a
        # line-for-line copy of torchvision's (so default behavior is
        # bit-identical to vanilla), but the ``box_head_forward`` slot lets
        # future variants (DoubleHead etc.) override the box-head pipeline
        # by subclassing CustomRoIHeads without touching Architecture.
        self.roi_heads = m.roi_heads
        self.roi_heads.__class__ = CustomRoIHeads

        # Box branch — only the ROI pool is exposed at top level; box_head /
        # box_predictor are called via ``self.roi_heads.box_head_forward``.
        self.box_roi_pool = m.roi_heads.box_roi_pool

        # Mask branch.
        self.mask_roi_pool  = m.roi_heads.mask_roi_pool
        self.mask_head      = m.roi_heads.mask_head
        self.mask_predictor = m.roi_heads.mask_predictor

        # Optional feature-level attention module. Baseline configs omit
        # cfg.MODEL["attention"], leaving this as Identity and preserving
        # vanilla behavior. CBAM experiments apply the module to each FPN
        # feature map after the backbone and before RPN/ROI heads.
        self.attention = self._build_attention(cfg)

        # Optional RoI-level attention module for the mask branch. Baseline
        # configs omit cfg.MODEL["mask_attention"], leaving this as Identity.
        # The surrounding mask-branch call sequence follows torchvision's
        # RoIHeads.forward; this hook is inserted between mask_roi_pool and
        # mask_head.
        self.mask_attention = self._build_mask_attention(cfg)
        self.mask_correction = self._build_mask_correction(cfg)
        self.residual_mask_refinement = self._build_residual_mask_refinement(cfg)

        # Optional proposal-overlap refinement for box regression. Baseline
        # configs omit cfg.MODEL["box_overlap_refinement"], leaving this unset
        # and preserving the original box head.
        self.roi_heads._box_overlap_refinement = self._build_box_overlap_refinement(cfg)

        # Optional RoI relation refinement for box-branch spatial features.
        # Baseline configs omit cfg.MODEL["box_roi_relation_refinement"], leaving
        # this unset and preserving the original RoI box feature path.
        self.roi_heads._box_roi_relation_refinement = self._build_box_roi_relation_refinement(cfg)

        # External orchestrator (training class) sets this to a dict during
        # val passes (with epoch / log_dir / diagnostics flags); diagnostic
        # functions read it and stay no-op when it's None.
        self.state = None

        # DEBUG flags fire on every forward (train + val); cfg.DEBUG is the
        # dict from the cfg, fpn_input / fpn_output toggle the two prints
        # in forward.
        self.debug = getattr(cfg, "DEBUG", {}) or {}

        # Mask loss is selected once at build-time via cfg; default keeps the
        # vanilla BCE path. forward() never touches the registry — it only
        # calls self.mask_loss_fn.
        mask_loss_name = cfg.MODEL.get("mask_loss", "maskrcnn_loss")
        self.mask_loss_fn = MASK_LOSSES.get(mask_loss_name)
        if self.mask_loss_fn is None:
            raise KeyError(
                f"cfg.MODEL['mask_loss']='{mask_loss_name}' not in MASK_LOSSES. "
                f"available: {MASK_LOSSES.keys()}"
            )

        # Box-head NMS is selected once at build-time via cfg; default is
        # "vanilla" (= torchvision batched_nms), bit-identical to upstream.
        # The function + kwargs get attached to the roi_heads instance so
        # CustomRoIHeads.postprocess_detections can call them at the NMS line.
        nms_cfg = cfg.MODEL.get("box_nms", {"type": "vanilla"})
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

        # Box-head regression loss (cfg-swappable). Default "smooth_l1" is
        # bit-identical to upstream torchvision fastrcnn_loss. cfg can switch
        # to "giou" (or any future variant in BOX_REG_LOSSES).
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

    @property
    def gated_residual_refinement(self):
        """Old attribute used by earlier analysis scripts."""
        return self.residual_mask_refinement

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # earlier checkpoints used this module prefix
        old_prefix = prefix + "gated_residual_refinement."
        new_prefix = prefix + "residual_mask_refinement."
        for key in list(state_dict):
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                if new_key in state_dict:
                    error_msgs.append(f"Duplicate refinement weights: {key} and {new_key}")
                else:
                    state_dict[new_key] = state_dict.pop(key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def _build_residual_mask_refinement(self, cfg):
        refinement_cfg = cfg.MODEL.get(
            "residual_mask_refinement", cfg.MODEL.get("gated_residual_refinement")
        )
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

        # ── transform / backbone / rpn ─────────────────────────────────
        images, targets = self.transform(x, targets)
        if self.debug.get("fpn_input"):
            print(f"[DEBUG/fpn_input] images.tensors={tuple(images.tensors.shape)} "
                  f"image_sizes={images.image_sizes}")
        features = self.backbone(images.tensors)
        if self.debug.get("fpn_output"):
            shapes = ", ".join(f"{k}={tuple(v.shape)}" for k, v in features.items())
            print(f"[DEBUG/fpn_output] {shapes}")
        features = self.attention(features)
        proposals, proposal_losses = self.rpn(images, features, targets)
        accumulate_rpn_coverage(proposals, targets, self.state)

        # ── RoI: training-time sampling ────────────────────────────────
        if self.training:
            proposals, matched_idxs, labels, regression_targets = \
                self.roi_heads.select_training_samples(proposals, targets)
        else:
            matched_idxs = labels = regression_targets = None

        # ── BOX branch ─────────────────────────────────────────────────
        box_features = self.box_roi_pool(features, proposals, images.image_sizes)
        if self.debug.get("roi_align_output"):
            print(f"[DEBUG/roi_align_output] box_features={tuple(box_features.shape)}")
        class_logits, box_regression = self.roi_heads.box_head_forward(
            box_features,
            proposals,
            matched_idxs=matched_idxs,
            labels=labels,
        )
        if self.debug.get("box_output"):
            print(f"[DEBUG/box_output] class_logits={tuple(class_logits.shape)} "
                  f"box_regression={tuple(box_regression.shape)}")

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
        # ── MASK branch ────────────────────────────────────────────────
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

        if self.debug.get("mask_input"):
            n_per = [int(p.shape[0]) for p in mask_proposals]
            print(f"[DEBUG/mask_input] proposal_count_per_image={n_per}  "
                  f"(mode={'train' if self.training else 'eval'})")
        mask_features = self.mask_roi_pool(features, mask_proposals, images.image_sizes)
        if self.debug.get("mask_input"):
            print(f"[DEBUG/mask_input] mask_features (post-RoIAlign)={tuple(mask_features.shape)}")
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
        if self.debug.get("mask_input"):
            print(f"[DEBUG/mask_input] mask_features (post-mask-attention)={tuple(mask_features.shape)}")
        mask_features = self.mask_head(mask_features)
        mask_logits   = self.mask_predictor(mask_features)
        # Baseline shape check: [num_rois, num_classes, 28, 28].
        # print(f"[SHAPE/mask_predictor] mask_logits={tuple(mask_logits.shape)}")
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

        # ── postprocess (eval only) ────────────────────────────────────
        if not self.training:
            result = self.transform.postprocess(
                result, images.image_sizes, original_image_sizes,
            )

        if self.training:
            return {**proposal_losses, **detector_losses}
        return result
