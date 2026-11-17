#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MAIN="${RELEASE_ROOT}/backbones/AdaGCN/main.py"
DATA_ROOT="${DATA_ROOT:-${RELEASE_ROOT}/Citation_mat}"
SEEDS="1 2 3 4 5"
read -r -a COMMON_EXTRA <<< "${COMMON_ARGS:-}"
read -r -a SEED_LIST <<< "${SEEDS}"

TASKS=(A_to_C A_to_D C_to_A C_to_D D_to_A D_to_C)
SOURCES=(acmv9 acmv9 citationv1 citationv1 dblpv7 dblpv7)
TARGETS=(citationv1 dblpv7 acmv9 dblpv7 acmv9 citationv1)

for seed in "${SEED_LIST[@]}"; do
  for index in "${!TASKS[@]}"; do
    echo "RUN_TASK,method=becat,task=${TASKS[$index]},seed=${seed}"
    python "${MAIN}" \
      --single_pair=true \
      --source="${SOURCES[$index]}" \
      --target="${TARGETS[$index]}" \
      --data_root="${DATA_ROOT}" \
      --seed="${seed}" \
      --split_seed=42 \
      --use_becat=true \
      --sparse_gt_to_critic=false \
      "${COMMON_EXTRA[@]}" \
      --latent_topk_k=10 \
      --latent_topk_tau=0.5 \
      --sparse_gt_curv_bias_beta=0.05 \
      --sparse_gt_residual_alpha=0.1 \
      --lambda_latent_bec=0.001 \
      --lambda_latent_bec_stat=0.1 \
      --sparse_gt_fourier_m=4 \
      --sparse_gt_eta_abs=1.0 \
      --sparse_gt_eta_fourier=0.1 \
      --latent_bec_warmup_epochs=100 \
      --sparse_gt_curv_bias_warmup=100 \
      --gnn=gcn
  done
done
