#!/usr/bin/env bash
set -euo pipefail

RELEASE_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
DATA_ROOT=${1:-${DATA_ROOT:-}}
: "${DATA_ROOT:?Usage: run_becat.sh /absolute/path/to/Citation}"
PYTHON_BIN=${PYTHON_BIN:-python}

task_names=(A_to_C A_to_D C_to_A C_to_D D_to_A D_to_C)
sources=(ACMv9 ACMv9 Citationv1 Citationv1 DBLPv7 DBLPv7)
targets=(Citationv1 DBLPv7 ACMv9 DBLPv7 ACMv9 Citationv1)

cd "$RELEASE_ROOT"
for task_index in "${!task_names[@]}"; do
    for seed in 23 202 11 27 201; do
        echo "RUN,task=${task_names[$task_index]},seed=$seed,split_seed=42,method=BECAT"
        "$PYTHON_BIN" -m backbones.SpecReg.main_specReg \
            --data_root "$DATA_ROOT" \
            --source "${sources[$task_index]}" \
            --target "${targets[$task_index]}" \
            --seed "$seed" \
            --split_seed 42 \
            --use_becat \
            --lambda_latent_bec 0.001 \
            --lambda_latent_bec_stat 0.1 \
            --latent_topk_k 10 \
            --latent_topk_tau 0.5 \
            --latent_bec_sample_size 128 \
            --sparse_gt_attention_k 10 \
            --sparse_gt_curv_bias_beta 0.05 \
            --sparse_gt_residual_alpha 0.1 \
            --sparse_gt_num_heads 2 \
            --sparse_gt_num_layers 1 \
            --sparse_gt_dropout 0.3 \
            --sparse_gt_curv_bias_mode hybrid \
            --sparse_gt_bias_norm scale \
            --sparse_gt_fourier_m 4 \
            --sparse_gt_fourier_sigma 1.0 \
            --sparse_gt_fourier_seed 1234 \
            --sparse_gt_eta_abs 1.0 \
            --sparse_gt_eta_fourier 0.1
        # latent_bec_use_mmd=False: --latent_bec_use_mmd is intentionally absent.
    done
done
