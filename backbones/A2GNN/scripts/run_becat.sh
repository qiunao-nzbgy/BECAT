#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${1:-}}"
SEEDS="200 201 202 203 204"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:1}"
ENCODER_DEVICE="${ENCODER_DEVICE:-${DEVICE}}"
GEOMETRY_DEVICE="${GEOMETRY_DEVICE:-${ENCODER_DEVICE}}"

if [[ -z "${DATA_ROOT}" ]]; then
  echo "Set DATA_ROOT to the external Citation directory." >&2
  exit 2
fi
TASKS=(
  "A_to_C ACMv9 Citationv1"
  "A_to_D ACMv9 DBLPv7"
  "C_to_A Citationv1 ACMv9"
  "C_to_D Citationv1 DBLPv7"
  "D_to_A DBLPv7 ACMv9"
  "D_to_C DBLPv7 Citationv1"
)

cd "${RELEASE_ROOT}"
for task in "${TASKS[@]}"; do
  read -r task_name source target <<< "${task}"
  for seed in ${SEEDS}; do
    echo "Running ${task_name} BECAT seed=${seed} split_seed=42"
    "${PYTHON_BIN}" -m backbones.A2GNN.main \
      --data_root "${DATA_ROOT}" \
      --source "${source}" \
      --target "${target}" \
      --seed "${seed}" \
      --split_seed 42 \
      --device "${DEVICE}" \
      --encoder_device "${ENCODER_DEVICE}" \
      --geometry_device "${GEOMETRY_DEVICE}" \
      --lr 0.005 \
      --weight_decay 0.005 \
      --dropout_ratio 0.5 \
      --nhid 128 \
      --patience 100 \
      --epochs 200 \
      --weight 5 \
      --source_pnum 0 \
      --target_pnum 30 \
      --use_becat true \
      --lambda_becat 1.0 \
      --latent_topk_k 10 \
      --latent_topk_tau 0.5 \
      --lambda_latent_bec 0.001 \
      --lambda_latent_bec_stat 0.1 \
      --latent_bec_use_mmd false \
      --latent_bec_sample_size 128 \
      --sparse_gt_num_heads 2 \
      --sparse_gt_num_layers 1 \
      --sparse_gt_dropout 0.3 \
      --sparse_gt_curv_bias_beta 0.05 \
      --sparse_gt_curv_bias_mode hybrid \
      --sparse_gt_fourier_m 4 \
      --sparse_gt_fourier_sigma 1.0 \
      --sparse_gt_fourier_seed 1234 \
      --sparse_gt_eta_abs 1.0 \
      --sparse_gt_eta_fourier 0.1 \
      --sparse_gt_bias_norm scale \
      --sparse_gt_attention_k 10 \
      --sparse_gt_residual_alpha 0.1
  done
done
