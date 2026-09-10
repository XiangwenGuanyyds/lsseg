# LSSeg

Instance segmentation experiments for occlusion-robust pig segmentation.

## Environment Setup

The experiments use Python 3.10, PyTorch 2.7.1 and torchvision 0.22.1 on
Linux/WSL, with CUDA 11.8.

```bash
conda create -n lsseg python=3.10 -y
conda activate lsseg
```

Install PyTorch for an NVIDIA GPU with a CUDA 11.8-compatible driver:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
```

For CPU use, install this build instead:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
```

Install the remaining dependencies and check the environment:

```bash
python -m pip install -r requirements.txt
python -m pip check
./lsseg --help
```

`./lsseg` uses Python from the active environment. The model downloads and
caches ImageNet ResNet-50 weights on first use.

## Project Structure

| Path | Purpose |
| --- | --- |
| `main.py` | Direct training and test entry point |
| `configs/` | Experiment configurations |
| `models/` | Architecture, heads and losses |
| `data_loaders/` | Dataset parsing, transforms and data loaders |
| `runners/` | Training and test loops |
| `evaluation/` | Evaluation metrics and analysis |
| `optim/` | Optimiser builders |
| `tools/` | Dataset viewing, annotation, plotting and visualisation |
| `dataset/` | Original data, prepared splits and consensus labels |
| `outputs/` | Results grouped by experiment |

## Dataset Setup

Download `dataset.zip` from [tubCloud](https://tubcloud.tu-berlin.de/s/LDbXc9o3yyAF8fi).
Password: `gxwlsseg2026`.

The ZIP file includes the original data, prepared splits and occlusion consensus
labels. The source dataset is available on
[Roboflow Universe](https://universe.roboflow.com/nbing/pig-uebpf-eh8lw).

Extract `dataset.zip` into the project root:

```bash
python -m zipfile -e /path/to/dataset.zip .
```

After extraction, the `dataset/` folder should sit beside `main.py`.
The paths used by the experiments are:

```text
lsseg/
├── main.py
├── configs/
└── dataset/
    ├── raw/pigs/pig/
    │   ├── train/
    │   ├── valid/
    │   └── test/
    └── per_scene/mix/
        ├── train.coco.json
        ├── val.coco.json
        ├── test.coco.json
        └── occlusion_review/consensus/
            ├── occluded.val.json
            ├── not_occluded.val.json
            ├── ambiguous.val.json
            ├── occluded.test.json
            ├── not_occluded.test.json
            ├── ambiguous.test.json
            └── summary.json
```

The prepared `mix` splits contain 500 training, 100 validation and 100 test
images. They are ready to use with the supplied consensus labels.

Browse images by split and consensus group:

```bash
python tools/view_dataset.py mix
```

The viewer opens in a browser and prints its address in the terminal.

## Train and Test

See [Experiment Configurations](configs/CONFIGURATIONS.md) for the available
configurations. Run commands from the project root.

For training and testing in one command, see [Batch Scripts](#batch-scripts).

Baseline:

```bash
./lsseg train baseline_seed0 --config baseline --seed 0
./lsseg test baseline_seed0
```

Final model:

```bash
./lsseg train final_model_seed0 --config dice_residual_mask_refinement --seed 0
./lsseg test final_model_seed0 --config final_model
```

Training includes validation with standard NMS. Testing is a separate command;
the Soft-NMS configurations are test-only. Repeat with seeds 1 and 2 and distinct
experiment names for three-seed experiments.

Testing loads the named experiment's training checkpoint. Without `--config`,
it uses the saved `training/config.json`; an explicit `--config` selects the
test configuration. Use `--result-name` to keep a separate set of test results:

```bash
./lsseg test baseline_seed0 --config gaussian_soft_nms --result-name gaussian_soft_nms_seed0
```

Reusing an experiment name for training overwrites its training outputs.
Repeating a test replaces the test and analysis outputs under its result name.

## Output Directories

| Output directory | Contents |
| --- | --- |
| `outputs/<experiment>/training/` | Checkpoint, saved configuration, metadata and training/validation metrics |
| `outputs/<experiment>/test/` | Predictions, ground truth, AP metrics, test configuration and metadata |
| `outputs/<experiment>/analysis/` | Pipeline analysis, pixel metrics, mask-error analysis and figures |

## Plot Curves

```bash
python tools/plot.py baseline_seed0
python tools/plot.py baseline_seed0 final_model_seed0 --metrics all
python tools/plot.py baseline_seed0 final_model_seed0 --type val --groups occluded --metrics AP AP75
```

Single-experiment plots are saved to `outputs/<experiment>/analysis/figures/`.
Comparisons use timestamped filenames in `local_record/plots/`.
Use `--output path/to/figure.png` to set the output path.

## Visualise Predictions

```bash
python tools/visualize.py --list-experiments
python tools/visualize.py baseline_seed0 --list-ids
python tools/visualize.py baseline_seed0 --image-ids 667 --list-ann-ids
python tools/visualize.py baseline_seed0 --ann-ids 8274
python tools/visualize.py baseline_seed0 --image-ids 123 --view both
python tools/visualize.py baseline_seed0 final_model_seed0 --image-ids 123 456
```

Replace the example IDs with those returned by the list commands.
`--ann-ids` produces crops for selected ground-truth instances;
`--view both` saves full images and instance crops. Use `--score-threshold`
to change the default prediction score threshold of 0.5, or `--device cpu`
to run on CPU.

The tool uses the saved test configuration and checkpoint, falling back to the
training configuration and checkpoint when no test configuration exists.

Single-experiment images are saved to
`outputs/<experiment>/analysis/visualizations/`. Comparisons use timestamped
filenames in `local_record/visualizations/`. Both plotting tools print the saved
paths; repeating a single-experiment plot or view replaces the corresponding image.

## Batch Scripts

Each script trains and tests one seed before moving to the next.
Run from the project root with the project environment active.

| Script | Training configuration | Test configuration | Results |
| --- | --- | --- | --- |
| `run_baseline.sh` | `baseline` | `baseline` (standard NMS) | `outputs/baseline_seed<N>/` |
| `run_final_model.sh` | `dice_residual_mask_refinement` | `final_model` (Thresholded Gaussian Soft-NMS) | `outputs/final_model_seed<N>/` |

```bash
# Default: seed 0
./run_baseline.sh
./run_final_model.sh

# One selected seed
./run_baseline.sh --seeds 2
./run_final_model.sh --seeds 2

# Multiple seeds
./run_baseline.sh --seeds 0 1 2
./run_final_model.sh --seeds 0 1 2
```

Each script prints the experiment list before starting. It stops if any selected
experiment directory already exists or a training or test command fails.
