#!/usr/bin/env bash
# Wait for GPU headroom, run val90 holdout FT, then TTA submit.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
TAG="${BATCH_TAG:-20260519}"
MIN_FREE_MIB="${MIN_FREE_MIB:-18000}"
POLL_SEC="${POLL_SEC:-120}"
LOG_FT="logs/hc/ablette_hc_shc_s42_val90_holdout_${TAG}.log"
FT_PID_FILE="logs/hc/ablette_hc_shc_s42_val90_holdout_${TAG}.pid"
CKPT_VAL90="checkpoints/track_a/hc/shc_n4_s42_ft_val90_holdout.pt"

if [[ -f "${CKPT_VAL90}" ]]; then
  echo "[$(date -Is)] ${CKPT_VAL90} exists; skip FT" | tee -a "${LOG_FT}"
  exec env BATCH_TAG="${TAG}" bash scripts/ablette_hc_shc_val90_ft_then_submit.sh
fi

if [[ -f "${FT_PID_FILE}" ]] && kill -0 "$(cat "${FT_PID_FILE}")" 2>/dev/null; then
  echo "[$(date -Is)] FT already running pid=$(cat "${FT_PID_FILE}")" | tee -a "${LOG_FT}"
  exec env BATCH_TAG="${TAG}" bash scripts/ablette_hc_shc_val90_ft_then_submit.sh
fi

while true; do
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "[$(date -Is)] GPU free ${free_mib} MiB (need >= ${MIN_FREE_MIB})" | tee -a "${LOG_FT}"
  if [[ "${free_mib}" -ge "${MIN_FREE_MIB}" ]]; then
    break
  fi
  sleep "${POLL_SEC}"
done

echo "=== START $(date -Iseconds) val90 holdout FT ===" | tee -a "${LOG_FT}"
nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "${PY}" -u -m smth2smth.pipelines.train \
  experiment=track_a_hc_ablation_shc_val90_holdout track=a seed=42 \
  >> "${LOG_FT}" 2>&1 &
echo $! > "${FT_PID_FILE}"
echo "[$(date -Is)] launched FT pid=$(cat "${FT_PID_FILE}")" | tee -a "${LOG_FT}"

exec env BATCH_TAG="${TAG}" bash scripts/ablette_hc_shc_val90_ft_then_submit.sh
