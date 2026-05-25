#!/usr/bin/env bash
# Wait for val90 FT (if running), then test submit with best TTA.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
TAG="${BATCH_TAG:-20260519}"
CKPT_VAL90="${CKPT_VAL90:-checkpoints/track_a/hc/shc_n4_s42_ft_val90_holdout.pt}"
LOG_FT="logs/hc/ablette_hc_shc_s42_val90_holdout_${TAG}.log"
LOG_SUB="logs/hc/ablette_hc_shc_s42_submit_val90_${TAG}.log"
SUB_OUT="submissions/track_a_hc_shc_s42_val90_holdout_${TAG}.csv"
FT_PID_FILE="logs/hc/ablette_hc_shc_s42_val90_holdout_${TAG}.pid"

_ft_running() {
  pgrep -f 'smth2smth\.pipelines\.train.*track_a_hc_ablation_shc_val90_holdout' >/dev/null 2>&1
}

if _ft_running; then
  echo "[wait $(date -Is)] val90 holdout FT still running" | tee -a "${LOG_FT}"
  while _ft_running; do sleep 60; done
  echo "[wait $(date -Is)] val90 holdout FT finished" | tee -a "${LOG_FT}"
fi

if [[ ! -f "${CKPT_VAL90}" ]]; then
  echo "[$(date -Is)] ERROR: missing ${CKPT_VAL90} after FT" | tee -a "${LOG_FT}"
  exit 1
fi

echo "[$(date -Is)] submit ckpt=${CKPT_VAL90} -> ${SUB_OUT}" | tee "${LOG_SUB}"
env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "${PY}" -u -m smth2smth.pipelines.submit \
  track=a experiment=track_a_hc_ablation_shc \
  "training.checkpoint_path=${CKPT_VAL90}" \
  "dataset.submission_output=${SUB_OUT}" \
  training.tta=true training.tta_flip=true \
  'training.tta_scales=[0.857,1.0,1.143]' \
  2>&1 | tee -a "${LOG_SUB}"
echo "[$(date -Is)] submit done -> ${SUB_OUT}" | tee -a "${LOG_SUB}"
