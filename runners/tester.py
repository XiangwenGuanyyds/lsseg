"""Load one checkpoint and write a complete test output for an experiment.

Mirrors the eval portion of Trainer.validate() but standalone — no
train_loader, optimizer, scheduler, or criterion. Test predictions, metrics,
ground truth, and pipeline analysis are written to
their canonical subdirectories under ``outputs/<experiment>/``.
"""

import os
import shutil

import torch

from data_loaders.build import build_dataloader, build_dataset
from evaluation.coco_ap_report import write_coco_ap_report
from evaluation.pipeline_analysis import (
    write_postprocessing_recall,
    write_rpn_coverage,
)
from evaluation.segmentation_ap import MaskAPEvaluator, write_mask_ap_log
from models.build import build_model
from utils.experiment_metadata import write_test_provenance


class Tester:

    def __init__(self, cfg, paths, weights, config_name=None, command=None):
        if not os.path.exists(weights):
            raise FileNotFoundError(f"weights not found: {weights}")

        self.cfg = cfg
        self.paths = paths
        self.exp_dir = str(paths.test)
        self.weights = weights
        self.config_name = config_name
        self.command = command
        self.device = torch.device(cfg.DEVICE)

        self.test_dataset = build_dataset(cfg, mode="test")
        self.test_loader  = build_dataloader(self.test_dataset, cfg, mode="test")

        self.model = build_model(cfg).to(self.device)
        state_dict = torch.load(weights, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"[checkpoint] loaded complete trained model state: {weights}")

        # A repeated test invalidates every derived test and analysis output.
        shutil.rmtree(paths.test, ignore_errors=True)
        shutil.rmtree(paths.analysis, ignore_errors=True)
        paths.test.mkdir(parents=True, exist_ok=True)
        paths.test_metrics.mkdir(parents=True, exist_ok=True)
        paths.ground_truth.mkdir(parents=True, exist_ok=True)
        paths.pipeline_analysis.mkdir(parents=True, exist_ok=True)
        self.log_dir = str(paths.test_metrics)

    def test(self):
        # state for in-forward diagnostics; epoch=0 since test is one-shot.
        self.model.state = {
            "epoch":       0,
            "log_dir":     self.log_dir,
            "pipeline_analysis_dir": str(self.paths.pipeline_analysis),
            "diagnostics": getattr(self.cfg, "DIAGNOSTICS", {}),
        }
        mask_ap = MaskAPEvaluator(
            ann_file         = self.test_dataset.parser.ann_file,
            eval_class_ids   = self.cfg.EVAL_CLASS_IDS,
            occluded_ann_ids = getattr(self.test_dataset, "_occluded_ids", set()),
            not_occluded_ann_ids = getattr(
                self.test_dataset, "_not_occluded_ids", None
            ),
            category_remap   = getattr(self.cfg, "CATEGORY_REMAP", None),
            debug            = getattr(self.cfg, "DEBUG", {}).get("mask_ap_input", False),
        )

        self.model.eval()
        with torch.no_grad():
            for images, targets in self.test_loader:
                inputs = [img.to(self.device) for img in images]
                model_targets = [
                    {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                     for k, v in t.items()}
                    for t in targets
                ]
                predictions = self.model(inputs, model_targets)
                image_ids = [int(t["image_id"].item()) for t in targets]
                mask_ap.update_batch(predictions, image_ids)

        # dump BEFORE compute() — COCOeval mutates GT segmentation into bytes
        # RLE counts in-place, which breaks json serialization downstream.
        mask_ap.dump(self.exp_dir, str(self.paths.ground_truth))
        result = mask_ap.compute()
        self.model.state["mask_ap_result"] = result
        write_rpn_coverage(self.model.state, 0)
        write_postprocessing_recall(self.model.state, 0)
        write_mask_ap_log(self.model.state, 0)
        report_path, _ = write_coco_ap_report(
            self.paths.test,
            self.paths.test_metrics,
        )
        print(f"wrote {report_path}")
        config_path, metadata_path = write_test_provenance(
            cfg=self.cfg,
            exp_dir=self.exp_dir,
            checkpoint_path=self.weights,
            config_name=self.config_name,
            command=self.command,
            test_annotation_path=self.test_dataset.parser.ann_file,
            test_occlusion_path=self.test_dataset.occlusion_label_file,
            test_not_occluded_path=self.test_dataset.not_occluded_label_file,
            test_image_dir=getattr(self.test_dataset.parser, "img_dir", None),
        )
        print(f"wrote {config_path}")
        print(f"wrote {metadata_path}")
        self.model.state = None

        self._print_summary(result)

    def _print_summary(self, result):
        exp_name = self.paths.experiment
        print(f"\n========== Test Results: {exp_name} ==========")
        print(f"{'group':16s}{'AP[.5:.95]':>14s}{'AP50':>10s}{'AP75':>10s}")
        for grp in ("overall", "occluded", "not_occluded"):
            d = result.get(grp)
            if d is None:
                print(f"{grp:16s}{'─':>14s}{'─':>10s}{'─':>10s}  (无数据)")
                continue
            print(f"{grp:16s}{d['AP']:>14.4f}{d['AP50']:>10.4f}{d['AP75']:>10.4f}")
        print("=" * 50)
