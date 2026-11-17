#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DATA_ROOT:-}" ]]; then
  echo "Set DATA_ROOT to the external Citation directory." >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PAIRALIGN_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SEED_ARGS=(1 2 3 4 5)
read -r -a EXTRA_ARGS <<< "${PAIRALIGN_ARGS:-}"

TASKS=(A_to_C A_to_D C_to_A C_to_D D_to_A D_to_C)
SOURCES=(ACMv9 ACMv9 Citationv1 Citationv1 DBLPv7 DBLPv7)
TARGETS=(Citationv1 DBLPv7 ACMv9 DBLPv7 ACMv9 Citationv1)

for index in "${!TASKS[@]}"; do
  echo "RUN,method=PairAlign,task=${TASKS[$index]}"
  "${PYTHON_BIN}" "${PAIRALIGN_DIR}/train_da.py" \
    --data_root "${DATA_ROOT}" \
    --src_name "${SOURCES[$index]}" \
    --tgt_name "${TARGETS[$index]}" \
    --seed "${SEED_ARGS[@]}" \
    --split_seed 42 \
    "${EXTRA_ARGS[@]}"
done
