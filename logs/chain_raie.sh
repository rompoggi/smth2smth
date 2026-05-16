#!/usr/bin/env bash
# Chain launcher for SSL.md Track A, codename "raie".
# Waits for Phase 1 (VideoMAE pretrain) to exit cleanly with the encoder
# checkpoint on disk, then launches Phase 2 (attentive probe fine-tune).
# All output goes to logs/ssl_raie_chain.log so it can be tailed independently.

set -uo pipefail

REPO_ROOT="/Data/romain.poggi/smth2smth"
UV="/users/eleves-a/2021/romain.poggi/.local/bin/uv"
CODENAME="raie"

cd "${REPO_ROOT}"

PRETRAIN_PID_FILE="logs/ssl_${CODENAME}_pretrain.pid"
ENCODER_PATH="checkpoints/track_a/ssl/${CODENAME}_encoder.pt"
PRETRAIN_LOG="logs/ssl_${CODENAME}_pretrain.log"
FT_LOG="logs/ssl_${CODENAME}_finetune.log"
FT_PID_FILE="logs/ssl_${CODENAME}_finetune.pid"

ts() { date -Iseconds; }

if [[ ! -f "${PRETRAIN_PID_FILE}" ]]; then
  echo "[$(ts)] [chain] no pretrain pid file at ${PRETRAIN_PID_FILE}; aborting"
  exit 1
fi

PRETRAIN_PID="$(cat "${PRETRAIN_PID_FILE}")"
echo "[$(ts)] [chain] watching pretrain pid=${PRETRAIN_PID} (uv wrapper)"

# Poll until the wrapper process is gone. We also check the python child via
# pgrep so we don't fire Phase 2 while the real worker is still running.
while kill -0 "${PRETRAIN_PID}" 2>/dev/null \
   || pgrep -f "smth2smth.pipelines.pretrain_videomae" >/dev/null; do
  sleep 60
done

echo "[$(ts)] [chain] pretrain process exited"

# Confirm encoder checkpoint exists and the success line was logged.
if [[ ! -f "${ENCODER_PATH}" ]]; then
  echo "[$(ts)] [chain] ERROR: encoder checkpoint missing at ${ENCODER_PATH}"
  echo "[$(ts)] [chain] last 40 pretrain log lines:"
  tail -n 40 "${PRETRAIN_LOG}"
  exit 2
fi

if ! grep -q "wrote encoder checkpoint" "${PRETRAIN_LOG}"; then
  echo "[$(ts)] [chain] ERROR: 'wrote encoder checkpoint' line not found in ${PRETRAIN_LOG}"
  echo "[$(ts)] [chain] last 40 pretrain log lines:"
  tail -n 40 "${PRETRAIN_LOG}"
  exit 3
fi

echo "[$(ts)] [chain] encoder ok ($(stat -c '%s bytes' "${ENCODER_PATH}")); launching Phase 2 fine-tune"

nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src "${UV}" run python -u \
    -m smth2smth.pipelines.train \
    experiment=track_a_ssl_finetune_${CODENAME} track=a \
    > "${FT_LOG}" 2>&1 &

FT_PID=$!
echo "${FT_PID}" > "${FT_PID_FILE}"
echo "[$(ts)] [chain] fine-tune launched, pid=${FT_PID}, log=${FT_LOG}"
