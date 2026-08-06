#!/usr/bin/env bash
# Train N independent causal seeds for R1 ensemble (grok_recs).
# Isolates seed only — same config otherwise (smooth_lookahead=0).
#
# Usage (from repo root, on a GPU box with data):
#   bash model_training/realtime/train_ensemble_seeds.sh 8
#   bash model_training/realtime/train_ensemble_seeds.sh 3 results/trained_models/ensemble_causal_la0
#
# Each run ≈ 6.4 h on RTX 3090. Do NOT bundle other hyperparameter changes.

set -euo pipefail

N_SEEDS="${1:-8}"
OUT_ROOT="${2:-results/trained_models/ensemble_causal_la0}"
BASE_CFG="${3:-results/trained_models/causal_la0/checkpoint/args.yaml}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
if [[ -x .venv/bin/python ]]; then
  PYTHON=".venv/bin/python"
fi

for ((s=0; s<N_SEEDS; s++)); do
  echo "=== Training seed ${s} → ${OUT_ROOT}/seed_${s} ==="
  "$PYTHON" model_training/realtime/launch_seed_train.py \
    --seed "$s" \
    --base "$BASE_CFG" \
    --out_root "$OUT_ROOT"
done

echo "Done. Ensemble members:"
find "$OUT_ROOT" -name best_checkpoint 2>/dev/null | sort || true
