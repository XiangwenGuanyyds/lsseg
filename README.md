# LSSeg

Code for *Occlusion-Robust Instance Segmentation of Pigs in Livestock Environments*.
The final model combines ResNet-50-FPN Mask R-CNN with Dice loss, Residual Mask
Refinement and Thresholded Gaussian Soft-NMS.

## Installation

The experiments use Python 3.10, PyTorch 2.7.1 and torchvision 0.22.1 on Linux/WSL
with CUDA 11.8. Run the following commands from the project root:

```bash
conda create -n lsseg python=3.10 -y
conda activate lsseg
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

The model downloads ImageNet-pretrained ResNet-50 weights on first use.

## Dataset

Download `dataset.zip` from [tubCloud](https://tubcloud.tu-berlin.de/s/LDbXc9o3yyAF8fi).
Password: `gxwlsseg2026`.

Extract it into the project root so that `dataset/` sits beside `main.py`:

```bash
python -m zipfile -e /path/to/dataset.zip .
```

The dataset includes the original images from
[Roboflow Universe](https://universe.roboflow.com/nbing/pig-uebpf-eh8lw),
the prepared `mix` split (500 training, 100 validation and 100 test images),
and occlusion labels agreed upon in two annotation rounds.

```text
dataset/
  raw/pigs/pig/
    train/
    valid/
    test/
  per_scene/mix/
    train.coco.json
    val.coco.json
    test.coco.json
    occlusion_review/consensus/
```

## Experiment Configurations

Each name below corresponds to `configs/<name>_cfg.py` and can be passed to
`--config`. The baseline uses BCE mask loss and standard NMS.

| Training configuration | Experiment |
| --- | --- |
| `baseline` | ResNet-50-FPN Mask R-CNN baseline |
| `dice` | Baseline + Dice loss |
| `residual_mask_refinement` | Baseline + Residual Mask Refinement |
| `dice_residual_mask_refinement` | Baseline + Dice loss + Residual Mask Refinement; used to train the final model |
| `dice_residual_mask_refinement_without_uncertainty_weight` | Dice + refinement with the uncertainty weight removed |

These configurations use standard NMS for validation. The following test
configurations apply Soft-NMS to the corresponding trained model:

| Test configuration | Experiment | Training configuration |
| --- | --- | --- |
| `gaussian_soft_nms` | Baseline + Gaussian Soft-NMS | `baseline` |
| `thresholded_gaussian_soft_nms` | Baseline + Thresholded Gaussian Soft-NMS | `baseline` |
| `final_model` | Dice + refinement + Thresholded Gaussian Soft-NMS | `dice_residual_mask_refinement` |
| `final_model_without_refinement` | Final model with refinement removed | `dice` |
| `final_model_without_uncertainty_weight` | Final model with the uncertainty weight removed | `dice_residual_mask_refinement_without_uncertainty_weight` |

## Training

Train and test the final model with seeds 0, 1 and 2:

```bash
./run_final_model.sh --seeds 0 1 2
```

The script runs each seed in turn and saves the results as `final_model_seed0`,
`final_model_seed1` and `final_model_seed2`. It uses standard NMS for validation
and Thresholded Gaussian Soft-NMS for testing. Existing experiment directories
must be renamed or moved before running it again.

For the baseline, use `./run_baseline.sh --seeds 0 1 2`. Both scripts default to
seed 0 when `--seeds` is omitted.

To train an individual experiment:

```bash
./lsseg train dice_seed0 --config dice --seed 0
```

Add `--validation-loss` to the training command to record validation loss each epoch.

## Evaluation

Evaluate a trained final model on the test set:

```bash
./lsseg test final_model_seed0 --config final_model
```

This loads the checkpoint from `outputs/final_model_seed0/training/` and writes
test results under the same experiment. The three-seed script already performs
this step for each seed. To display the saved bbox and segmentation AP results:

```bash
./lsseg ap final_model_seed0
```

For a separate post-processing experiment, keep the results under a new name:

```bash
./lsseg test baseline_seed0 --config gaussian_soft_nms --result-name gaussian_soft_nms_seed0
```

## Results

Each experiment has its own directory under `outputs/`:

| Directory | Contents |
| --- | --- |
| `outputs/<experiment>/training/` | Checkpoint, configuration, metadata and training/validation logs |
| `outputs/<experiment>/test/` | Predictions, ground truth, AP results, test configuration and metadata |
| `outputs/<experiment>/analysis/` | RPN coverage, recall retention, pixel metrics and mask-error analysis |

Repeated training or testing under the same name overwrites the corresponding
results. Use a new experiment name to keep previous runs.
