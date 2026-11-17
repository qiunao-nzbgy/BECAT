#!/usr/bin/env bash
set -euo pipefail

RELEASE_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
DATA_ROOT=${1:?Usage: run_baseline.sh /absolute/path/to/Citation}
PYTHON_BIN=${PYTHON_BIN:-python}
CUDA_DEVICE=${CUDA_DEVICE:-0}

task_names=(A_to_C A_to_D C_to_A C_to_D D_to_A D_to_C)
sources=(ACMv9 ACMv9 Citationv1 Citationv1 DBLPv7 DBLPv7)
targets=(Citationv1 DBLPv7 ACMv9 DBLPv7 ACMv9 Citationv1)

cd "$RELEASE_ROOT"
for task_index in "${!task_names[@]}"; do
    for seed in 1 2 3 4 5; do
        echo "RUN,task=${task_names[$task_index]},seed=$seed,method=baseline"
        "$PYTHON_BIN" -m backbones.GRADE.main \
            --data_root "$DATA_ROOT" \
            --source "${sources[$task_index]}" \
            --target "${targets[$task_index]}" \
            --seed "$seed" \
            --split_seed 42 \
            --n_hidden 8 \
            --n_layers 2 \
            --epochs 100 \
            --patience 20 \
            --lr 0.05 \
            --weight_decay 0.0001 \
            --dropout 0.0 \
            --cuda "$CUDA_DEVICE"
    done
done
