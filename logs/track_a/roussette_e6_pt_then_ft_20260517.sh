#!/usr/bin/env bash
# Wait for E6 MAE pretrain (pid file), then launch champion FT @256.
set -euo pipefail

REPO_ROOT="/Data/romain.poggi/smth2smth"
BATCH_TAG="20260517"
PT_PID_FILE="${REPO_ROOT}/logs/roussette_e6_mae_pretrain_224_${BATCH_TAG}.pid"
PT_LOG="${REPO_ROOT}/logs/roussette_e6_mae_pretrain_224_${BATCH_TAG}.log"
ENCODER="${REPO_ROOT}/checkpoints/track_a/ssl/e6_encoder.pt"
FT_LOG="${REPO_ROOT}/logs/roussette_e6_ft_champion_256_${BATCH_TAG}.log"
FT_PID_FILE="${REPO_ROOT}/logs/roussette_e6_ft_champion_256_${BATCH_TAG}.pid"
PY="${REPO_ROOT}/.venv/bin/python"

log() { echo "[e6-chain] $(date -Is) $*"; }

if [[ ! -f "${PT_PID_FILE}" ]]; then
  log "ERROR: missing PT pid file ${PT_PID_FILE}"
  exit 1
fi

PT_PID="$(cat "${PT_PID_FILE}")"
log "waiting for MAE pretrain pid=${PT_PID}"

while kill -0 "${PT_PID}" 2>/dev/null; do
  sleep 60
done

log "pretrain process exited; checking log and encoder"
if ! grep -q "epoch 100/100 avg loss" "${PT_LOG}"; then
  log "ERROR: PT log missing 'epoch 100/100 avg loss' — not launching FT"
  exit 1
fi
if [[ ! -s "${ENCODER}" ]]; then
  log "ERROR: encoder missing or empty: ${ENCODER}"
  exit 1
fi

if [[ -f "${FT_PID_FILE}" ]] && kill -0 "$(cat "${FT_PID_FILE}")" 2>/dev/null; then
  log "FT already running pid=$(cat "${FT_PID_FILE}") — skip"
  exit 0
fi

log "starting champion FT @256"
cd "${REPO_ROOT}"
nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "${PY}" -u -m smth2smth.pipelines.train \
  experiment=track_a_ssl_finetune_e6 track=a \
  >> "${FT_LOG}" 2>&1 &
FT_PID=$!
echo "${FT_PID}" > "${FT_PID_FILE}"
log "FT started pid=${FT_PID} log=${FT_LOG}"
