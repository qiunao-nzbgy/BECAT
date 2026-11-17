#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${1:-}}"
SEEDS="1 2 3 4 5"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

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
    echo "Running ${task_name} baseline seed=${seed} split_seed=42"
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
      --lambda_f 0.0001
  done
done
