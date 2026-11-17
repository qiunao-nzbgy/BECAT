#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKBONE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-${BACKBONE_DIR}/data}"
LOG_ROOT="${LOG_ROOT:-${BACKBONE_DIR}/logs/baseline}"
GPU_IDS="${GPU_IDS:-1,7}"
mkdir -p "${LOG_ROOT}"

run_task() {
  local task="$1"
  local source="$2"
  local target="$3"

  for seed in 1 2 3 4 5; do
    local log_file="${LOG_ROOT}/${task}_seed${seed}.log"
    echo "BASELINE task=${task} source=${source} target=${target} seed=${seed} split_seed=42 log=${log_file}"
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" "${BACKBONE_DIR}/main.py" \
      --data_root "${DATA_ROOT}" \
      --source "${source}" \
      --target "${target}" \
      --epoch 100 \
      --seed "${seed}" \
      --split_seed 42 \
      >"${log_file}" 2>&1
  done
}

run_task A_to_C ACMv9 Citationv1
run_task A_to_D ACMv9 DBLPv7
run_task C_to_A Citationv1 ACMv9
run_task C_to_D Citationv1 DBLPv7
run_task D_to_A DBLPv7 ACMv9
run_task D_to_C DBLPv7 Citationv1
