#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${1:-}}"
SEEDS="1 2 3 4 5"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

LATENT_TOPK_K="12"
LATENT_TOPK_TAU="0.7779185611540578"
LAMBDA_LATENT_BEC="0.00037119056439157197"
LAMBDA_LATENT_BEC_STAT="0.09540542632066311"
LATENT_BEC_SAMPLE_SIZE="${LATENT_BEC_SAMPLE_SIZE:-128}"
SPARSE_GT_NUM_HEADS="${SPARSE_GT_NUM_HEADS:-2}"
SPARSE_GT_NUM_LAYERS="${SPARSE_GT_NUM_LAYERS:-1}"
SPARSE_GT_DROPOUT="0.09208245914658439"
SPARSE_GT_CURV_BIAS_BETA="0.004245689393588735"
SPARSE_GT_CURV_BIAS_MODE="${SPARSE_GT_CURV_BIAS_MODE:-hybrid}"
SPARSE_GT_FOURIER_M="${SPARSE_GT_FOURIER_M:-4}"
SPARSE_GT_FOURIER_SIGMA="${SPARSE_GT_FOURIER_SIGMA:-1.0}"
SPARSE_GT_ETA_ABS="${SPARSE_GT_ETA_ABS:-1.0}"
SPARSE_GT_ETA_FOURIER="${SPARSE_GT_ETA_FOURIER:-0.1}"
SPARSE_GT_BIAS_NORM="${SPARSE_GT_BIAS_NORM:-scale}"
SPARSE_GT_ATTENTION_K="12"
SPARSE_GT_RESIDUAL_ALPHA="0.005"

if [[ -z "${DATA_ROOT}" ]]; then
  echo "Set DATA_ROOT to the external Citation_mat directory." >&2
  exit 2
fi
TASKS=(
  "A_to_C acmv9 citationv1"
  "A_to_D acmv9 dblpv7"
  "C_to_A citationv1 acmv9"
  "C_to_D citationv1 dblpv7"
  "D_to_A dblpv7 acmv9"
  "D_to_C dblpv7 citationv1"
)

cd "${RELEASE_ROOT}"
for task in "${TASKS[@]}"; do
  read -r task_name source target <<< "${task}"
  for seed in ${SEEDS}; do
    echo "Running ${task_name} BECAT seed=${seed} split_seed=42"
    "${PYTHON_BIN}" -m backbones.ASN.main \
      --data_root "${DATA_ROOT}" \
      --data_src "${source}" \
      --data_trg "${target}" \
      --seed "${seed}" \
      --split_seed 42 \
      --cuda "${CUDA_DEVICE}" \
      --lr 3e-2 \
      --n_epoch 2000 \
      --eval_interval 10 \
      --hidden 128 \
      --gfeat 16 \
      --nfeat 6775 \
      --dropout 0.5 \
      --classes 5 \
      --lambda_d 0.5 \
      --lambda_r 1 \
      --lambda_f 0.0001 \
      --use_becat \
      --latent_topk_k "${LATENT_TOPK_K}" \
      --latent_topk_tau "${LATENT_TOPK_TAU}" \
      --lambda_latent_bec "${LAMBDA_LATENT_BEC}" \
      --lambda_latent_bec_stat "${LAMBDA_LATENT_BEC_STAT}" \
      --latent_bec_sample_size "${LATENT_BEC_SAMPLE_SIZE}" \
      --sparse_gt_num_heads "${SPARSE_GT_NUM_HEADS}" \
      --sparse_gt_num_layers "${SPARSE_GT_NUM_LAYERS}" \
      --sparse_gt_dropout "${SPARSE_GT_DROPOUT}" \
      --sparse_gt_curv_bias_beta "${SPARSE_GT_CURV_BIAS_BETA}" \
      --sparse_gt_curv_bias_mode "${SPARSE_GT_CURV_BIAS_MODE}" \
      --sparse_gt_fourier_m "${SPARSE_GT_FOURIER_M}" \
      --sparse_gt_fourier_sigma "${SPARSE_GT_FOURIER_SIGMA}" \
      --sparse_gt_eta_abs "${SPARSE_GT_ETA_ABS}" \
      --sparse_gt_eta_fourier "${SPARSE_GT_ETA_FOURIER}" \
      --sparse_gt_bias_norm "${SPARSE_GT_BIAS_NORM}" \
      --sparse_gt_attention_k "${SPARSE_GT_ATTENTION_K}" \
      --sparse_gt_residual_alpha "${SPARSE_GT_RESIDUAL_ALPHA}"
  done
done
