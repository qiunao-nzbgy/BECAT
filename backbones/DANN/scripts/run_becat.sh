#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${1:-}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
SEEDS="1 2 3 4 5"

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
    echo "Running ${task_name} DANN+BECAT seed=${seed} split_seed=42"
    "${PYTHON_BIN}" -m backbones.DANN.main \
      --data_root "${DATA_ROOT}" \
      --source "${source}" \
      --target "${target}" \
      --seed "${seed}" \
      --split_seed 42 \
      --device "${DEVICE}" \
      --epochs 1000 \
      --lr 0.001 \
      --weight_decay 0.0005 \
      --hidden_dim 128 \
      --dropout 0.5 \
      --lambda_domain 1.0 \
      --use_becat \
      --lambda_becat 0.001
  done
done
