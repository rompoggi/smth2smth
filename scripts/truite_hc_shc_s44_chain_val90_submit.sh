#!/usr/bin/env bash
# Wait for honest-FT Kaggle submit, run val90 holdout FT, then submit holdout ckpt.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BATCH_TAG="${BATCH_TAG:-20260519}"
PY="${REPO_ROOT}/.venv/bin/python"
SUBMIT1_PID_FILE="logs/hc/truite_hc_shc_s44_submit_${BATCH_TAG}.pid"
SUBMIT1_LOG="logs/hc/truite_hc_shc_s44_submit_${BATCH_TAG}.log"
FT_LOG="logs/hc/truite_hc_shc_s44_val90_holdout_${BATCH_TAG}.log"
FT_PID_FILE="logs/hc/truite_hc_shc_s44_val90_holdout_${BATCH_TAG}.pid"
SUBMIT2_LOG="logs/hc/truite_hc_shc_s44_val90_submit_${BATCH_TAG}.log"
SUBMIT2_PID_FILE="logs/hc/truite_hc_shc_s44_val90_submit_${BATCH_TAG}.pid"
CHAIN_LOG="logs/hc/truite_hc_shc_s44_chain_val90_${BATCH_TAG}.log"
HOLDOUT_CKPT="checkpoints/track_a/hc/shc_n4_s44_ft_val90_holdout.pt"
SUBMIT2_OUT="submissions/track_a_hc_shc_s44_val90_holdout_${BATCH_TAG}.csv"

log() { echo "[hc-chain $(date -Is)] $*" | tee -a "$CHAIN_LOG"; }

if [[ ! -x "$PY" ]]; then
  echo "Missing .venv — run: uv sync" >&2
  exit 1
fi

if [[ -n "${SUBMIT1_PID:-}" ]]; then
  WAIT_PID="$SUBMIT1_PID"
elif [[ -f "$SUBMIT1_PID_FILE" ]]; then
  WAIT_PID="$(cat "$SUBMIT1_PID_FILE")"
else
  log "ERROR: set SUBMIT1_PID or create $SUBMIT1_PID_FILE before launching chain"
  exit 1
fi
log "Waiting for honest submit pid=$WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 30
done
log "Honest submit pid $WAIT_PID exited."

log "Launching val90 holdout FT: track_a_hc_ablation_shc_val90_holdout"
export PYTHONUNBUFFERED=1 PYTHONPATH=src
nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$PY" -u -m smth2smth.pipelines.train \
  experiment=track_a_hc_ablation_shc_val90_holdout \
  track=a seed=44 \
  >> "$FT_LOG" 2>&1 &
FT_PID=$!
echo "$FT_PID" > "$FT_PID_FILE"
log "Started holdout FT pid=$FT_PID log=$FT_LOG"

while kill -0 "$FT_PID" 2>/dev/null; do
  sleep 60
done
log "Holdout FT pid $FT_PID exited."

if [[ ! -s "$HOLDOUT_CKPT" ]]; then
  log "ERROR: missing holdout checkpoint $HOLDOUT_CKPT"
  tail -40 "$FT_LOG" | tee -a "$CHAIN_LOG" || true
  exit 1
fi

log "Launching holdout submit ckpt=$HOLDOUT_CKPT -> $SUBMIT2_OUT"
nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$PY" -u -m smth2smth.pipelines.submit \
  experiment=track_a_hc_ablation_shc_val90_holdout track=a seed=44 \
  training.checkpoint_path="${REPO_ROOT}/${HOLDOUT_CKPT}" \
  'training.tta_scales=[1.0]' \
  dataset.submission_output="${REPO_ROOT}/${SUBMIT2_OUT}" \
  >> "$SUBMIT2_LOG" 2>&1 &
SUBMIT2_PID=$!
echo "$SUBMIT2_PID" > "$SUBMIT2_PID_FILE"
log "Started holdout submit pid=$SUBMIT2_PID log=$SUBMIT2_LOG"
wait "$SUBMIT2_PID" || true
log "Holdout submit finished (see $SUBMIT2_LOG)."
