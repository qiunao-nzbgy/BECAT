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

LAMBDA_GEOMETRY="1e-3"
GEO_LAMBDA_STAT="0.1"
GEO_BETA="0.015656782335209403"
GEO_RESIDUAL_ALPHA="0.15"
GEO_TAU="0.8564065894099969"
GEO_K="8"
GEO_DROPOUT="0.3043708477144173"
GEO_NUM_HEADS="2"

TASKS=(A_to_C A_to_D C_to_A C_to_D D_to_A D_to_C)
SOURCES=(ACMv9 ACMv9 Citationv1 Citationv1 DBLPv7 DBLPv7)
TARGETS=(Citationv1 DBLPv7 ACMv9 DBLPv7 ACMv9 Citationv1)

for index in "${!TASKS[@]}"; do
  echo "RUN,method=PairAlign+BECAT,task=${TASKS[$index]}"
  "${PYTHON_BIN}" "${PAIRALIGN_DIR}/train_da.py" \
    --data_root "${DATA_ROOT}" \
    --src_name "${SOURCES[$index]}" \
    --tgt_name "${TARGETS[$index]}" \
    --seed "${SEED_ARGS[@]}" \
    --split_seed 42 \
    --use_becat \
    "${EXTRA_ARGS[@]}" \
    --lambda_geometry "${LAMBDA_GEOMETRY}" \
    --geo_lambda_stat "${GEO_LAMBDA_STAT}" \
    --geo_beta "${GEO_BETA}" \
    --geo_residual_alpha "${GEO_RESIDUAL_ALPHA}" \
    --geo_tau "${GEO_TAU}" \
    --geo_k "${GEO_K}" \
    --geo_dropout "${GEO_DROPOUT}" \
    --geo_num_heads "${GEO_NUM_HEADS}"
done
