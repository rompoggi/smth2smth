#!/usr/bin/env bash
# Val90 holdout classifier-only FT, then test submission (TTA from checkpoint cfg).
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
TAG="${BATCH_TAG:-$(date +%Y%m%d_%H%M%S)}"
EXPT="track_b_vjepa2_hfclf_8f_lora_r16_val90_holdout_clf"
CKPT_FT="checkpoints/track_b/vitl_fpc16ssv2_8f_lora_r16_val90_holdout_clf.pt"
LOG_FT="logs/track_b_vjepa2_hfclf_8f_lora_r16_val90_holdout_clf_${TAG}.log"
LOG_SUB="logs/track_b_vjepa2_hfclf_8f_lora_r16_val90_holdout_clf_submit_${TAG}.log"
SUB_OUT="submissions/track_b_vjepa2_hfclf_8f_lora_r16_val90_holdout_clf_${TAG}.csv"
FT_PID_FILE="logs/track_b_vjepa2_hfclf_8f_lora_r16_val90_holdout_clf_${TAG}.pid"

_ft_running() {
  pgrep -f "smth2smth\.pipelines\.train.*${EXPT}" >/dev/null 2>&1
}

if [[ "${SKIP_FT:-0}" != "1" ]]; then
  if _ft_running; then
    echo "[wait $(date -Is)] val90 holdout clf FT already running" | tee -a "${LOG_FT}"
    while _ft_running; do sleep 60; done
  else
    echo "[$(date -Is)] starting val90 holdout clf FT -> ${CKPT_FT}" | tee "${LOG_FT}"
    env PYTHONUNBUFFERED=1 PYTHONPATH=src \
      "${PY}" -u -m smth2smth.pipelines.train \
      track=b "experiment=${EXPT}" \
      2>&1 | tee -a "${LOG_FT}"
  fi
fi

if [[ ! -f "${CKPT_FT}" ]]; then
  echo "[$(date -Is)] ERROR: missing ${CKPT_FT} after FT" | tee -a "${LOG_FT}"
  exit 1
fi

echo "[$(date -Is)] submit ckpt=${CKPT_FT} -> ${SUB_OUT}" | tee "${LOG_SUB}"
env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "${PY}" -u -m smth2smth.pipelines.submit \
  track=b experiment=track_b_vjepa2_hfclf_8f_lora_r16 \
  "training.checkpoint_path=${CKPT_FT}" \
  "dataset.submission_output=${SUB_OUT}" \
  2>&1 | tee -a "${LOG_SUB}"
echo "[$(date -Is)] submit done -> ${SUB_OUT}" | tee -a "${LOG_SUB}"
