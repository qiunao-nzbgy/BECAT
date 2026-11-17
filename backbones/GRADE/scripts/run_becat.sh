#!/usr/bin/env bash
set -euo pipefail

RELEASE_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
DATA_ROOT=${1:?Usage: run_becat.sh /absolute/path/to/Citation}
PYTHON_BIN=${PYTHON_BIN:-python}
CUDA_DEVICE=${CUDA_DEVICE:-0}

task_names=(A_to_C A_to_D C_to_A C_to_D D_to_A D_to_C)
sources=(ACMv9 ACMv9 Citationv1 Citationv1 DBLPv7 DBLPv7)
targets=(Citationv1 DBLPv7 ACMv9 DBLPv7 ACMv9 Citationv1)

cd "$RELEASE_ROOT"
for task_index in "${!task_names[@]}"; do
    for seed in 1 2 3 4 5; do
        echo "RUN,task=${task_names[$task_index]},seed=$seed,method=BECAT"
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
            --cuda "$CUDA_DEVICE" \
            --use_becat \
            --becat_latent_topk_k 8 \
            --becat_sparse_gt_attention_k 8 \
            --becat_latent_topk_tau 1.0 \
            --becat_lambda_latent_bec 0.005 \
            --becat_lambda_latent_bec_stat 0.1 \
            --becat_sparse_gt_residual_alpha 0.15 \
            --becat_sparse_gt_curv_bias_beta 0.05 \
            --becat_latent_bec_sample_size 128
    done
done
