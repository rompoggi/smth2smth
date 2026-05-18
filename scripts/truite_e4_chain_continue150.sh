#!/usr/bin/env bash
# Wait for the in-flight E4 SGDR 90-ep job to exit, then launch continue150.
# Does not signal or stop the phase-1 process.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BATCH_TAG="${BATCH_TAG:-20260516}"
PY="${REPO_ROOT}/.venv/bin/python"
PHASE1_PID_FILE="logs/truite_e4_supervised_sgdr90_${BATCH_TAG}_resume.pid"
PHASE1_LOG="logs/truite_e4_supervised_sgdr90_${BATCH_TAG}.log"
PHASE2_LOG="logs/truite_e4_continue150_${BATCH_TAG}.log"
PHASE2_PID_FILE="logs/truite_e4_continue150_${BATCH_TAG}.pid"
CHAIN_LOG="logs/truite_e4_chain_continue150_${BATCH_TAG}.log"
LAST_CKPT="checkpoints/track_a/e4_ft.last.pt"

log() { echo "[chain $(date -Is)] $*" | tee -a "$CHAIN_LOG"; }

if [[ ! -x "$PY" ]]; then
  echo "Missing .venv — run: uv sync" >&2
  exit 1
fi

if [[ ! -f "$PHASE1_PID_FILE" ]]; then
  log "ERROR: missing $PHASE1_PID_FILE"
  exit 1
fi

WAIT_PID="$(cat "$PHASE1_PID_FILE")"
log "Waiting for phase-1 train pid=$WAIT_PID (no signals sent)."
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 120
done
log "Phase-1 pid $WAIT_PID exited."

if ! test -s "$LAST_CKPT"; then
  log "ERROR: $LAST_CKPT missing or empty after phase-1 exit."
  exit 1
fi

log "Launching phase-2: track_a_e4_tsm_continue150 (epochs 91-150)."
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$PY" -u -m smth2smth.pipelines.train \
  experiment=track_a_e4_tsm_continue150 track=a \
  "training.resume_from=${REPO_ROOT}/${LAST_CKPT}" \
  >> "$PHASE2_LOG" 2>&1 &
PHASE2_PID=$!
echo "$PHASE2_PID" > "$PHASE2_PID_FILE"
log "Started phase-2 pid=$PHASE2_PID log=$PHASE2_LOG pidfile=$PHASE2_PID_FILE"
