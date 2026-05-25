#!/usr/bin/env bash
# Wait for mHC s43 honest FT (60-epoch run), then launch val90 holdout continuation.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BATCH_TAG="${BATCH_TAG:-20260519}"
PY="${REPO_ROOT}/.venv/bin/python"
WAIT_PID_FILE="${REPO_ROOT}/logs/hc/roussette_hc_mhc_s43_${BATCH_TAG}.pid"
FT_LOG="${REPO_ROOT}/logs/hc/roussette_hc_mhc_s43_${BATCH_TAG}.log"
HOLDOUT_LOG="${REPO_ROOT}/logs/hc/roussette_hc_mhc_s43_val90_holdout_${BATCH_TAG}.log"
HOLDOUT_PID_FILE="${REPO_ROOT}/logs/hc/roussette_hc_mhc_s43_val90_holdout_${BATCH_TAG}.pid"
BEST_CKPT="${REPO_ROOT}/checkpoints/track_a/hc/mhc_n4_sk3_s43_ft.pt"
CHAIN_LOG="${REPO_ROOT}/logs/hc/roussette_hc_mhc_s43_val90_holdout_chain_${BATCH_TAG}.log"

log() { echo "[mhc-s43-chain] $(date -Is) $*" | tee -a "$CHAIN_LOG"; }

if [[ ! -x "$PY" ]]; then
  echo "Missing .venv — run: uv sync" >&2
  exit 1
fi
if [[ ! -f "$WAIT_PID_FILE" ]]; then
  log "ERROR: missing FT pid file ${WAIT_PID_FILE}"
  exit 1
fi

FT_PID="$(cat "$WAIT_PID_FILE")"
log "waiting for honest FT pid=${FT_PID} (log=${FT_LOG})"

_honest_ft_running() {
  local pid="$1"
  kill -0 "${pid}" 2>/dev/null \
    || pgrep -f "smth2smth.pipelines.train experiment=track_a_hc_ablation_mhc track=a seed=43" >/dev/null 2>&1
}

while _honest_ft_running "${FT_PID}"; do
  sleep 60
  if [[ -f "${WAIT_PID_FILE}" ]]; then
    FT_PID="$(cat "${WAIT_PID_FILE}")"
  fi
done

log "honest FT process exited; checking log and best checkpoint"

if ! grep -qE "Done\.|Training finished" "${FT_LOG}"; then
  log "WARN: FT log missing 'Done.' — holdout launch may be premature; tail:"
  tail -n 20 "${FT_LOG}" | tee -a "$CHAIN_LOG"
fi
if [[ ! -s "${BEST_CKPT}" ]]; then
  log "ERROR: best checkpoint missing or empty: ${BEST_CKPT}"
  exit 2
fi

if [[ -f "${HOLDOUT_PID_FILE}" ]] && kill -0 "$(cat "${HOLDOUT_PID_FILE}")" 2>/dev/null; then
  log "val90 holdout already running pid=$(cat "${HOLDOUT_PID_FILE}") — skip"
  exit 0
fi

log "starting val90 holdout FT (experiment=track_a_hc_ablation_mhc_val90_holdout)"
nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "${PY}" -u -m smth2smth.pipelines.train \
  experiment=track_a_hc_ablation_mhc_val90_holdout track=a seed=43 \
  >> "${HOLDOUT_LOG}" 2>&1 &
HOLDOUT_PID=$!
echo "${HOLDOUT_PID}" > "${HOLDOUT_PID_FILE}"
log "holdout FT started pid=${HOLDOUT_PID} log=${HOLDOUT_LOG}"
