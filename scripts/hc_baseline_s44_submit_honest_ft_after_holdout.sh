#!/usr/bin/env bash
# After holdout FT finishes, submit from the honest 60-epoch baseline (train only +
# full official val for metrics — no val90 holdout in training). No TTA.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BATCH_TAG="${BATCH_TAG:-20260519}"
PY="${REPO_ROOT}/.venv/bin/python"
WAIT_PID_FILE="${REPO_ROOT}/logs/hc/thon_hc_baseline_s44_val90_holdout_${BATCH_TAG}.pid"
CKPT="${REPO_ROOT}/checkpoints/track_a/hc/baseline_s44_ft.pt"
OUT="${REPO_ROOT}/submissions/track_a_hc_baseline_s44_honest_ft_notta_${BATCH_TAG}.csv"
LOG="${REPO_ROOT}/logs/hc/thon_hc_baseline_s44_honest_ft_submit_notta_${BATCH_TAG}.log"
PID_OUT="${REPO_ROOT}/logs/hc/thon_hc_baseline_s44_honest_ft_submit_notta_${BATCH_TAG}.pid"

if [[ ! -x "$PY" ]]; then
  echo "Missing .venv — run: uv sync" >&2
  exit 1
fi
if [[ ! -s "$CKPT" ]]; then
  echo "Missing checkpoint: $CKPT" >&2
  exit 1
fi
if [[ ! -f "$WAIT_PID_FILE" ]]; then
  echo "Missing holdout train PID file: $WAIT_PID_FILE" >&2
  exit 1
fi

HOLDOUT_PID="$(cat "$WAIT_PID_FILE")"
echo "[$(date -Is)] Waiting for holdout FT PID ${HOLDOUT_PID}..." | tee "$LOG"
while kill -0 "${HOLDOUT_PID}" 2>/dev/null; do
  sleep 30
done
echo "[$(date -Is)] Holdout FT finished; starting honest-FT submit." | tee -a "$LOG"

export PYTHONUNBUFFERED=1 PYTHONPATH=src
env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$PY" -u -m smth2smth.pipelines.submit \
  track=a seed=44 \
  training.checkpoint_path="$CKPT" \
  training.tta=false \
  training.tta_flip=false \
  dataset.submission_output="$OUT" \
  2>&1 | tee -a "$LOG"

echo "[$(date -Is)] Wrote $OUT" | tee -a "$LOG"
