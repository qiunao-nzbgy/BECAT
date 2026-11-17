#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the external Citation directory}"
SEEDS="${SEEDS:-5 7 11 23 2026}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RELEASE_ROOT}/runs/baseline}"
ENCODER_DIM="${ENCODER_DIM:-16}"
EPOCHS="${EPOCHS:-200}"

TASK_NAMES=(A_to_C A_to_D C_to_A C_to_D D_to_A D_to_C)
SOURCES=(ACMv9 ACMv9 Citationv1 Citationv1 DBLPv7 DBLPv7)
TARGETS=(Citationv1 DBLPv7 ACMv9 DBLPv7 ACMv9 Citationv1)
read -r -a SEED_ARRAY <<< "${SEEDS}"

mkdir -p "${OUTPUT_ROOT}"
cd "${RELEASE_ROOT}"

for task_index in "${!TASK_NAMES[@]}"; do
  for seed in "${SEED_ARRAY[@]}"; do
    common_args=(
      --data_root "${DATA_ROOT}"
      --source "${SOURCES[task_index]}"
      --target "${TARGETS[task_index]}"
      --seed "${seed}"
      --split_seed 42
      --UDAGCN true
      --encoder_dim "${ENCODER_DIM}"
      --epochs "${EPOCHS}"
    )
    "${PYTHON_BIN}" -m backbones.UDAGCN.main "${common_args[@]}" \
      | tee "${OUTPUT_ROOT}/${TASK_NAMES[task_index]}_seed${seed}.log"
  done
done
