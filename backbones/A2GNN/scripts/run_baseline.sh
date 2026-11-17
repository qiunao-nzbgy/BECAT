#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${1:-}}"
SEEDS="200 201 202 203 204"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:1}"

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
    echo "Running ${task_name} baseline seed=${seed} split_seed=42"
    "${PYTHON_BIN}" -m backbones.A2GNN.main \
      --data_root "${DATA_ROOT}" \
      --source "${source}" \
      --target "${target}" \
      --seed "${seed}" \
      --split_seed 42 \
      --device "${DEVICE}" \
      --lr 0.005 \
      --weight_decay 0.005 \
      --dropout_ratio 0.5 \
      --nhid 128 \
      --patience 100 \
      --epochs 200 \
      --weight 5 \
      --source_pnum 0 \
      --target_pnum 30 \
      --use_becat false
  done
done
