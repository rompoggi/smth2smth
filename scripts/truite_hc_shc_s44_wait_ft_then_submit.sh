#!/usr/bin/env bash
# Wait for val90 holdout FT, then run Kaggle submit on the holdout checkpoint.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BATCH_TAG="${BATCH_TAG:-20260519}"
PY="${REPO_ROOT}/.venv/bin/python"
FT_PID_FILE="logs/hc/truite_hc_shc_s44_val90_holdout_${BATCH_TAG}.pid"
SUBMIT_LOG="logs/hc/truite_hc_shc_s44_val90_submit_${BATCH_TAG}.log"
SUBMIT_PID_FILE="logs/hc/truite_hc_shc_s44_val90_submit_${BATCH_TAG}.pid"
HOLDOUT_CKPT="checkpoints/track_a/hc/shc_n4_s44_ft_val90_holdout.pt"
SUBMIT_OUT="submissions/track_a_hc_shc_s44_val90_holdout_${BATCH_TAG}.csv"

log() { echo "[hc-submit-chain $(date -Is)] $*"; }

if [[ -n "${FT_PID:-}" ]]; then
  WAIT_PID="$FT_PID"
elif [[ -f "$FT_PID_FILE" ]]; then
  WAIT_PID="$(cat "$FT_PID_FILE")"
else
  log "ERROR: set FT_PID or create $FT_PID_FILE"
  exit 1
fi

log "Waiting for holdout FT pid=$WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done
log "Holdout FT pid $WAIT_PID exited."

if [[ ! -s "$HOLDOUT_CKPT" ]]; then
  log "ERROR: missing $HOLDOUT_CKPT"
  exit 1
fi

log "Submit holdout ckpt -> $SUBMIT_OUT"
export PYTHONUNBUFFERED=1 PYTHONPATH=src
nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$PY" -u -m smth2smth.pipelines.submit \
  experiment=track_a_hc_ablation_shc_val90_holdout \
  track=a seed=44 \
  training.checkpoint_path="${REPO_ROOT}/${HOLDOUT_CKPT}" \
  'training.tta_scales=[1.0]' \
  dataset.submission_output="${REPO_ROOT}/${SUBMIT_OUT}" \
  >> "$SUBMIT_LOG" 2>&1 &
echo $! > "$SUBMIT_PID_FILE"
log "Started holdout submit pid=$(cat "$SUBMIT_PID_FILE")"
