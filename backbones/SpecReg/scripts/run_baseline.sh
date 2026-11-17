#!/usr/bin/env bash
set -euo pipefail

RELEASE_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
DATA_ROOT=${1:-${DATA_ROOT:-}}
: "${DATA_ROOT:?Usage: run_baseline.sh /absolute/path/to/Citation}"
PYTHON_BIN=${PYTHON_BIN:-python}

task_names=(A_to_C A_to_D C_to_A C_to_D D_to_A D_to_C)
sources=(ACMv9 ACMv9 Citationv1 Citationv1 DBLPv7 DBLPv7)
targets=(Citationv1 DBLPv7 ACMv9 DBLPv7 ACMv9 Citationv1)

cd "$RELEASE_ROOT"
for task_index in "${!task_names[@]}"; do
    for seed in 23 202 11 27 201; do
        echo "RUN,task=${task_names[$task_index]},seed=$seed,split_seed=42,method=baseline"
        "$PYTHON_BIN" -m backbones.SpecReg.main_specReg \
            --data_root "$DATA_ROOT" \
            --source "${sources[$task_index]}" \
            --target "${targets[$task_index]}" \
            --seed "$seed" \
            --split_seed 42
    done
done
