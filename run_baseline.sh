#!/usr/bin/env bash

# Train the baseline, then test it with standard NMS for each seed.
# Run from the project root with the project environment active:
#   ./run_baseline.sh                 # seed 0
#   ./run_baseline.sh --seeds 2       # one seed
#   ./run_baseline.sh --seeds 0 1 2   # three seeds, run in this order
# Results: outputs/baseline_seed<N>/
# An existing experiment directory or a failed command stops the batch.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS=(0)

usage() {
    echo "Usage: ./${0##*/} [--seeds N ...]"
    echo "Train and test baseline. Default: seed 0."
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

echo "Baseline: training + testing"
echo "Training config: baseline"
echo "Test config: baseline"
for seed in "${SEEDS[@]}"; do
    echo "  baseline_seed${seed}"
done

for seed in "${SEEDS[@]}"; do
    experiment="baseline_seed${seed}"
    if [[ -e "$ROOT_DIR/outputs/$experiment" ]]; then
        echo "Refusing to overwrite existing experiment: outputs/$experiment" >&2
        exit 1
    fi
done

for seed in "${SEEDS[@]}"; do
    experiment="baseline_seed${seed}"
    echo "Training: $experiment"
    "$ROOT_DIR/lsseg" train "$experiment" --config baseline --seed "$seed"

    echo "Testing: $experiment"
    "$ROOT_DIR/lsseg" test "$experiment" --config baseline --seed "$seed"
done

echo "Completed baseline training and testing."
