"""Trainer: dataloader + model + loss + optimizer + scheduler + train loop + validate.

Per-epoch flow:
    train_one_epoch  -- per-step forward + backward
    validate         -- run model over val_loader, accumulate predictions into
                        MaskAPEvaluator, then write validation mask AP.

End of fit: save ``outputs/<experiment>/training/checkpoint.pth``.
"""

import csv
import os

import torch

from data_loaders.build import build_dataloader, build_dataset
from evaluation.segmentation_ap import MaskAPEvaluator, write_mask_ap_log
from models.build import build_model
from models.losses.build import build_loss
from optim import build_optimizer

_TRAIN_LOG_FILES = (
    "train_losses.csv",
    "validation_mask_ap.csv",
)

# Removed training-time diagnostics are still cleared when an experiment name
# is reused, so old CSVs cannot be mistaken for outputs of the new run.
_LEGACY_TRAIN_DIAGNOSTIC_FILES = (
    "mask_ap.csv",
    "rpn_coverage.csv",
    "box_iou_distribution.csv",
    "box_iou_distribution_prenms.csv",
    "box_candidate_counts.csv",
    "box_candidate_counts_prenms.csv",
    "box_center_offset.csv",
    "box_scale_error.csv",
    "box_mask_alignment.csv",
    "roi_coverage_loss.csv",
    "postprocessing_recall.csv",
)


def _remove_existing_outputs(output_dir, filenames):
    for filename in filenames:
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            os.remove(path)


class Trainer:

    def __init__(self, cfg, exp_dir):
        self.cfg = cfg
        self.exp_dir = exp_dir
        self.device = torch.device(cfg.DEVICE)
        self.epochs = cfg.NUM_EPOCHS

        train_dataset = build_dataset(cfg, mode="train")
        self.train_loader = build_dataloader(train_dataset, cfg, mode="train")
        self.val_dataset  = build_dataset(cfg, mode="val")
        self.val_loader   = build_dataloader(self.val_dataset, cfg, mode="val")

        self.model     = build_model(cfg).to(self.device)
        self.criterion = build_loss(cfg)
        self.optimizer = build_optimizer(cfg, self.model)
        self.scheduler = self._build_scheduler()

        self.log_dir = os.path.join(exp_dir, "metrics")
        os.makedirs(self.log_dir, exist_ok=True)
        checkpoint_path = os.path.join(exp_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        _remove_existing_outputs(
            self.log_dir,
            _TRAIN_LOG_FILES + _LEGACY_TRAIN_DIAGNOSTIC_FILES,
        )

    def _build_scheduler(self):
        steps_per_epoch = len(self.train_loader)
        if len(self.optimizer.param_groups) == 1:
            max_lr = self.cfg.OPTIMIZER["lr"] * 10
        else:
            max_lr = [g["lr"] * 10 for g in self.optimizer.param_groups]
        return torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        )

    def train(self):
        for epoch in range(self.epochs):
            train_means = self.train_one_epoch(epoch)
            val_ap      = self.validate(epoch)
            self._append_train_loss_csv(epoch, train_means)
            print(f"[epoch {epoch + 1}/{self.epochs}]  "
                  f"train_loss={train_means['total']:.4f}  val_AP={val_ap:.4f}")

        torch.save(self.model.state_dict(),
                   os.path.join(self.exp_dir, "checkpoint.pth"))

    def train_one_epoch(self, epoch):
        self.model.train()
        loss_sum, n_step = 0.0, 0
        comp_sums = {}                   # per-component running sum across the epoch
        for images, targets in self.train_loader:
            inputs = [img.to(self.device) for img in images]
            model_targets = [
                {"boxes":  t["boxes"].to(self.device),
                 "labels": t["labels"].to(self.device),
                 "masks":  t["masks"].to(self.device)}
                for t in targets
            ]
            loss_dict = self.model(inputs, model_targets)
            total_loss, weighted = self.criterion(loss_dict)
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            loss_sum += total_loss.item()
            for k, v in weighted.items():
                comp_sums[k] = comp_sums.get(k, 0.0) + v
            n_step += 1
        n = max(n_step, 1)
        means = {"total": loss_sum / n}
        for k, v in comp_sums.items():
            means[k] = v / n
        return means

    def _append_train_loss_csv(self, epoch, means):
        """Write one row per epoch to ``training/metrics/train_losses.csv``.

        Header is decided by the first epoch's loss component keys (``means``
        from train_one_epoch). Subsequent epochs reuse the original header;
        new keys appearing later are silently dropped, missing keys default
        to 0.0 — consistent with how the file was opened initially.
        """
        csv_path = os.path.join(self.log_dir, "train_losses.csv")
        if not os.path.exists(csv_path):
            comp_keys = sorted(k for k in means.keys() if k != "total")
            fields = ["epoch", "total"] + comp_keys
            with open(csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()
        else:
            with open(csv_path) as f:
                fields = next(csv.reader(f))
        row = {"epoch": epoch}
        for k in fields[1:]:
            row[k] = means.get(k, 0.0)
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

    def validate(self, epoch):
        self.model.state = {
            "epoch":       epoch,
            "log_dir":     self.log_dir,
            # Stage diagnostics are final-test outputs, not per-epoch
            # validation outputs.
            "diagnostics": {},
            "mask_ap_filename": "validation_mask_ap.csv",
        }
        mask_ap = MaskAPEvaluator(
            ann_file         = self.val_dataset.parser.ann_file,
            eval_class_ids   = self.cfg.EVAL_CLASS_IDS,
            occluded_ann_ids = getattr(self.val_dataset, "_occluded_ids", set()),
            not_occluded_ann_ids = getattr(
                self.val_dataset, "_not_occluded_ids", None
            ),
            category_remap   = getattr(self.cfg, "CATEGORY_REMAP", None),
            debug            = getattr(self.cfg, "DEBUG", {}).get("mask_ap_input", False),
        )

        self.model.eval()
        with torch.no_grad():
            for images, targets in self.val_loader:
                inputs = [img.to(self.device) for img in images]
                model_targets = [
                    {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                     for k, v in t.items()}
                    for t in targets
                ]
                predictions = self.model(inputs, model_targets)
                image_ids = [int(t["image_id"].item()) for t in targets]
                mask_ap.update_batch(predictions, image_ids)

        result = mask_ap.compute()
        self.model.state["mask_ap_result"] = result
        write_mask_ap_log(self.model.state, epoch)
        self.model.state = None
        return result["overall"]["AP"]
