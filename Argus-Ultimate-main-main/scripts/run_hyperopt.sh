#!/usr/bin/env bash
# Run Argus Optuna hyperopt — Push 51
# Usage: ./scripts/run_hyperopt.sh [N_TRIALS] [OUT_PATH]
#   N_TRIALS  : number of Optuna trials (default: $ARGUS_HYPEROPT_TRIALS or 100)
#   OUT_PATH  : output JSON path (default: optimization/best_params.json)

set -euo pipefail

N_TRIALS=${1:-${ARGUS_HYPEROPT_TRIALS:-100}}
OUT_PATH=${2:-optimization/best_params.json}

echo "[argus-hyperopt] Starting Optuna study: $N_TRIALS trials → $OUT_PATH"

python -m optimization.hyperopt_runner \
    --trials "$N_TRIALS" \
    --out "$OUT_PATH"

echo "[argus-hyperopt] Done. Best params written to $OUT_PATH"
