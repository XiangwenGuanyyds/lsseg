# LSSeg

Instance segmentation experiments for occlusion-robust pig segmentation.

## Environment Setup

The experiments use Python 3.10, PyTorch 2.7.1 and torchvision 0.22.1 on
Linux/WSL, with the CUDA 11.8 PyTorch build. The direct dependencies are pinned
in `requirements.txt`.

Create and activate an environment:

```bash
conda create -n lsseg python=3.10 -y
conda activate lsseg
```

For an NVIDIA GPU with a CUDA 11.8-compatible driver:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

For CPU use, run these installation commands instead:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The PyTorch commands follow the [official version-specific installation instructions](https://pytorch.org/get-started/previous-versions/#v271).
OpenCV uses the headless package; dataset viewing and annotation run in a web browser.

Check the installation:

```bash
python -m pip check
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('CUDA available:', torch.cuda.is_available())"
./lsseg --help
```

`./lsseg` uses `python` from the active environment. To select an interpreter
explicitly, set `LSSEG_PYTHON` to its executable path. Default model configurations
download the ImageNet ResNet-50 weights on first use and cache them locally.

## Project Structure

| Path | Purpose |
| --- | --- |
| `experiment.py` | Recommended train, test, metric, and experiment-info interface |
| `main.py` | Low-level training and test entry point |
| `configs/` | Registered experiment configurations |
| `models/` | Model architecture, heads, losses, and attention modules |
| `data_loaders/` | Dataset parsing, transforms, and data loaders |
| `runners/` | Training and test workflow orchestration |
| `evaluation/` | Evaluation metrics and pipeline analysis logic |
| `optim/` | Optimizer and scheduler builders |
| `tools/` | Standalone dataset, evaluation, visualization, and plotting tools |
| `dataset/` | Raw and constructed datasets |
| `outputs/` | Training, test, and analysis outputs grouped by experiment |
| `backups/` | Archived project backups |
| `tmp/` | Temporary inspection artifacts |

The configuration files remain flat under `configs/` because
`configs/__init__.py` automatically registers sibling `*_cfg.py` modules.

## Standard Workflow

```bash
# Train. The experiment saves its effective config and metadata.
./lsseg train \
  baseline_seed0 --config baseline --seed 0

# Test a trained experiment.
./lsseg test baseline_seed0

# Print the final bbox and segmentation COCO AP table.
./lsseg ap baseline_seed0

# Compute overall foreground pixel metrics from the saved test predictions.
./lsseg pixel-metrics baseline_seed0

# Read and validate all pipeline-analysis outputs produced during test.
./lsseg analyze pipeline baseline_seed0

# Compute all mask-error analyses from the saved test predictions.
./lsseg analyze mask-errors baseline_seed0

# Show the config, checkpoint, test, and analysis status.
./lsseg info baseline_seed0
```

To evaluate one checkpoint with a different test-time configuration, use a
separate result name:

```bash
./lsseg test \
  baseline_seed0 \
  --config gaussian_soft_nms \
  --result-name gaussian_soft_nms_seed0 \
  --seed 0
```

## Output Layout

All outputs belonging to an experiment share one directory:

```text
outputs/<experiment>/
├── training/
│   ├── config.json
│   ├── metadata.json
│   ├── checkpoint.pth
│   └── metrics/
│       ├── train_losses.csv
│       └── validation_mask_ap.csv
├── test/
│   ├── config.json
│   ├── metadata.json
│   ├── predictions.json
│   ├── ground_truth/
│   │   ├── overall.json
│   │   ├── occluded.json
│   │   └── not_occluded.json
│   └── metrics/
│       ├── mask_ap.csv
│       └── coco_ap_report.csv
└── analysis/
    ├── pipeline/
    │   ├── rpn_coverage.csv
    │   └── postprocessing_recall.csv
    ├── mask_errors/
    ├── pixel_metrics/
    └── figures/
```

RPN and post-processing outputs are collected during test. Mask-error and
pixel-level analyses are created only when their corresponding `./lsseg`
analysis command is run.

## Plot Experiment Curves

Read training CSV logs by experiment name:

```bash
python tools/plot.py baseline_seed0
python tools/plot.py baseline_seed0 dice_w16_seed0
python tools/plot.py baseline_seed0 dice_w16_seed0 --metrics all
python tools/plot.py baseline_seed0 dice_w16_seed0 --type val
python tools/plot.py baseline_seed0 dice_w16_seed0 --type val --groups occluded --metrics AP AP75
```

Each panel compares the same metric across experiments. Curves use all available
epochs. Loss columns absent from an experiment are skipped with a printed message.
For repeated epoch/group rows, the last recorded row is used.

Single-experiment figures go to `outputs/<experiment>/analysis/figures/`.
Their filenames are `training_loss.png` and `validation_mask_ap.png`;
rerunning replaces the same file. Multi-experiment comparisons go to
`local_record/plots/`, with a timestamp prepended to each filename.
Directories are created as needed, and the full saved path is printed.
Use `--output path/to/figure.png` to choose a different output path.

## Visualise Predictions

```bash
python tools/visualize.py --list-experiments
python tools/visualize.py baseline_seed0 --list-ids
python tools/visualize.py baseline_seed0 --image-ids 667 --list-ann-ids
python tools/visualize.py baseline_seed0 --ann-ids 8274
python tools/visualize.py baseline_seed0 dice_w16_seed0 --ann-ids 8274 8275
python tools/visualize.py baseline_seed0 --image-ids 123
python tools/visualize.py baseline_seed0 dice_w16_seed0 --image-ids 123 456
python tools/visualize.py baseline_seed0 --image-ids 123 --view both
```

Replace the example IDs with IDs printed by `--list-ids` or `--list-ann-ids`.
Omitting `--image-ids` with `--list-ann-ids` lists instances from all test images.
`--ann-ids` selects ground-truth annotation IDs and automatically finds their
images. It generates only the selected instance crops; multiple IDs can belong
to different images. All list commands avoid loading checkpoints and running
inference. The experiment list includes test-only results whose saved metadata
points to an existing training checkpoint.

The script reads the saved test configuration
and its checkpoint path; experiments without a test configuration use their
saved training configuration and checkpoint.

Each image shows ground truth beside predictions from the selected experiments.
`--view crops` produces local views around ground-truth boxes, using the same
crop for every experiment. `--view both` saves full images and crops. The default
prediction score threshold is 0.5; use `--score-threshold` to change it.

Single-experiment images go to `outputs/<experiment>/analysis/visualizations/`,
named by image ID and, for crops, annotation ID. Repeating the same view replaces
that image. Multi-experiment images go to `local_record/visualizations/` with
timestamped filenames. Missing output directories are created automatically;
the full path of each saved PNG is printed. Use `--device cpu` to run on CPU.

## Occlusion annotation

`tools/annotate_occlusion_round1.py` records the first round.
`tools/annotate_occlusion_round2.py` records the second round independently.
Their records are saved separately in `first_round/` and `v2/` under
`local_record/annotation_records/<scene>/`. Missing directories are created automatically.

```bash
python tools/annotate_occlusion_round1.py mix
python tools/annotate_occlusion_round2.py mix
python tools/annotate_occlusion_consensus.py mix
```

The last command checks that both rounds are complete and writes the occluded,
not-occluded and ambiguous ID lists to
`dataset/per_scene/mix/occlusion_review/consensus/`, as used by the experiment
configs. The annotation commands keep their output in the local record directory.

Browse images and consensus groups with the read-only viewer:

```bash
python tools/view_dataset.py mix
```

The page offers split and group filters. If consensus labels are missing, it
shows the annotation and consensus commands. The system assigns an available
local port, and the script opens the viewer in your browser. The address is also
printed in the terminal.
