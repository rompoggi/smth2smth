#!/usr/bin/env bash
# Generate Kaggle CSV from the current E4 trainval checkpoint (TTA from saved config).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BATCH_TAG="${BATCH_TAG:-20260516}"
PY="${REPO_ROOT}/.venv/bin/python"
CKPT="${REPO_ROOT}/checkpoints/track_a/e4_ft_trainval.pt"
OUT="${REPO_ROOT}/submissions/track_a_e4_trainval_ep6_${BATCH_TAG}.csv"
LOG="logs/truite_e4_trainval_submit_${BATCH_TAG}.log"

if [[ ! -x "$PY" ]]; then
  echo "Missing .venv — run: uv sync" >&2
  exit 1
fi
if [[ ! -s "$CKPT" ]]; then
  echo "Missing checkpoint: $CKPT" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1 PYTHONPATH=src

echo "[submit $(date -Is)] ckpt=$CKPT -> $OUT" | tee "$LOG"
env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$PY" -u -m smth2smth.pipelines.submit \
  track=a \
  training.checkpoint_path="$CKPT" \
  dataset.submission_output="$OUT" \
  2>&1 | tee -a "$LOG"

echo "[submit $(date -Is)] Wrote $OUT" | tee -a "$LOG"
