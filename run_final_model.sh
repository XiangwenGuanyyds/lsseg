#!/usr/bin/env bash

# Train Dice + Residual Mask Refinement, then test with Thresholded Gaussian
# Soft-NMS. Validation during training uses standard NMS.
# Run from the project root with the project environment active:
#   ./run_final_model.sh                 # seed 0
#   ./run_final_model.sh --seeds 2       # one seed
#   ./run_final_model.sh --seeds 0 1 2   # three seeds, run in this order
# Results: outputs/final_model_seed<N>/
# An existing experiment directory or a failed command stops the batch.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS=(0)

usage() {
    echo "Usage: ./${0##*/} [--seeds N ...]"
    echo "Train and test final model. Default: seed 0."
}

if [[ $# -gt 0 ]]; then
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --seeds)
            shift
            if [[ $# -eq 0 ]]; then
                echo "--seeds requires at least one seed." >&2
                exit 1
            fi
            SEEDS=("$@")
            ;;
        *)
            usage >&2
            exit 1
            ;;
    esac
fi

declare -A seen=()
for seed in "${SEEDS[@]}"; do
    if [[ ! "$seed" =~ ^(0|[1-9][0-9]{0,9})$ ]] || (( seed > 4294967295 )); then
        echo "Invalid seed: $seed. Use an integer from 0 to 4294967295." >&2
        exit 1
    fi
    if [[ -n "${seen[$seed]:-}" ]]; then
        echo "Repeated seed: $seed" >&2
        exit 1
    fi
    seen[$seed]=1
done

echo "Final model: training + testing"
echo "Training config: dice_residual_mask_refinement"
echo "Test config: final_model"
for seed in "${SEEDS[@]}"; do
    echo "  final_model_seed${seed}"
done

for seed in "${SEEDS[@]}"; do
    experiment="final_model_seed${seed}"
    if [[ -e "$ROOT_DIR/outputs/$experiment" ]]; then
        echo "Refusing to overwrite existing experiment: outputs/$experiment" >&2
        exit 1
    fi
done

for seed in "${SEEDS[@]}"; do
    experiment="final_model_seed${seed}"
    echo "Training: $experiment"
    "$ROOT_DIR/lsseg" train "$experiment" --config dice_residual_mask_refinement --seed "$seed"

    echo "Testing: $experiment"
    "$ROOT_DIR/lsseg" test "$experiment" --config final_model --seed "$seed"
done

echo "Completed final model training and testing."
