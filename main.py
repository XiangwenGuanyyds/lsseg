"""Run:
    python main.py --mode train --config <name> [--exp_name X] [--seed N]
    python main.py --mode test  --config <name> --weights <ckpt.pth> [--exp_name X]

Outputs are grouped under ``outputs/<experiment>/``. Training writes to the
``training/`` subdirectory and testing writes to ``test/``.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

import configs  # noqa: F401  (registers all configs at import time)
from runners import Tester, Trainer
from utils.experiment_utils import get_experiment_dir
from utils.output_paths import experiment_paths
from utils.registry import CONFIGS
from utils.experiment_metadata import write_train_provenance
from utils.saved_config import load_saved_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",     choices=["train", "test"], default="train")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="registered public config name")
    source.add_argument("--config-file", help="saved experiment config.json")
    p.add_argument("--exp_name", default=None)
    p.add_argument("--seed",     type=int, default=None)
    p.add_argument("--weights",  default=None,
                   help="path to checkpoint .pth (required for --mode test)")
    p.add_argument("--num_workers", type=int, default=None,
                   help="override cfg.NUM_WORKERS for this run")
    p.add_argument("--batch_size", type=int, default=None,
                   help="override cfg.BATCH_SIZE for this run")
    p.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="disable optional pipeline diagnostics during testing",
    )
    args = p.parse_args()

    if args.config_file:
        cfg, config_name = load_saved_config(args.config_file, Path(__file__).resolve().parent)
    else:
        config_name = args.config
        cfg = CONFIGS.get(config_name)
        if cfg is None:
            raise ValueError(f"Unknown --config '{config_name}'. Valid: {CONFIGS.keys()}")
    if args.mode == "train" and getattr(cfg, "TEST_ONLY", False):
        p.error("This config is for testing. Choose its corresponding training config.")

    if args.exp_name is not None:
        cfg.EXP_NAME = args.exp_name
    if args.seed is not None:
        cfg.SEED = args.seed
    if args.num_workers is not None:
        cfg.NUM_WORKERS = args.num_workers
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch_size must be at least 1")
        cfg.BATCH_SIZE = args.batch_size
    if args.no_diagnostics:
        cfg.DIAGNOSTICS = {}

    seed = getattr(cfg, "SEED", None)
    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))

    if args.mode == "train":
        exp_dir, _ = get_experiment_dir(cfg)
        command = [sys.executable, *sys.argv]
        write_train_provenance(
            cfg,
            exp_dir,
            config_name=config_name,
            command=command,
            status="running",
        )
        Trainer(cfg, exp_dir).train()
        write_train_provenance(
            cfg,
            exp_dir,
            config_name=config_name,
            command=command,
            status="completed",
            checkpoint_path=Path(exp_dir) / "checkpoint.pth",
        )
    else:
        # Resolve weights: explicit --weights wins; otherwise use the named
        # experiment's canonical training checkpoint.
        if args.weights:
            weights = Path(args.weights)
            if args.exp_name:
                exp_name = args.exp_name
            elif weights.parent.name == "training":
                exp_name = weights.parent.parent.name
            else:
                exp_name = weights.parent.name
        elif args.exp_name:
            exp_name = args.exp_name
            weights = experiment_paths(cfg.BASE_DIR, exp_name).checkpoint
        else:
            raise SystemExit("--mode test requires either --exp_name or --weights")
        paths = experiment_paths(cfg.BASE_DIR, exp_name)
        paths.test.mkdir(parents=True, exist_ok=True)
        Tester(
            cfg,
            paths,
            str(weights),
            config_name=config_name,
            command=[sys.executable, *sys.argv],
        ).test()


if __name__ == "__main__":
    main()
