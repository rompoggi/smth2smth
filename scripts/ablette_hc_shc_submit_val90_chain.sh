#!/usr/bin/env bash
# Honest-val submit -> 90% val holdout FT -> holdout submit (sequential).
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
TAG="${BATCH_TAG:-20260519}"
CKPT_HONEST="${CKPT_HONEST:-checkpoints/track_a/hc/shc_n4_s42_ft.pt}"
CKPT_VAL90="${CKPT_VAL90:-checkpoints/track_a/hc/shc_n4_s42_ft_val90_holdout.pt}"
SUB_HONEST="submissions/track_a_hc_shc_s42_honest_${TAG}.csv"
SUB_VAL90="submissions/track_a_hc_shc_s42_val90_holdout_${TAG}.csv"

LOG_SUB1="logs/hc/ablette_hc_shc_s42_submit_honest_${TAG}.log"
LOG_FT="logs/hc/ablette_hc_shc_s42_val90_holdout_${TAG}.log"
LOG_SUB2="logs/hc/ablette_hc_shc_s42_submit_val90_${TAG}.log"

run_submit() {
  local ckpt="$1" out="$2" log="$3"
  echo "[chain $(date -Is)] submit ckpt=$ckpt -> $out" | tee "$log"
  env PYTHONUNBUFFERED=1 PYTHONPATH=src \
    "$PY" -u -m smth2smth.pipelines.submit \
    track=a experiment=track_a_hc_ablation_shc \
    "training.checkpoint_path=${ckpt}" \
    "dataset.submission_output=${out}" \
    training.tta=true training.tta_flip=true \
    'training.tta_scales=[0.857,1.0,1.143]' \
    2>&1 | tee -a "$log"
  echo "[chain $(date -Is)] submit done -> $out" | tee -a "$log"
}

run_submit "$CKPT_HONEST" "$SUB_HONEST" "$LOG_SUB1"

echo "[chain $(date -Is)] starting val90 holdout FT" | tee "$LOG_FT"
env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$PY" -u -m smth2smth.pipelines.train \
  experiment=track_a_hc_ablation_shc_val90_holdout track=a seed=42 \
  2>&1 | tee -a "$LOG_FT"
echo "[chain $(date -Is)] val90 FT done" | tee -a "$LOG_FT"

run_submit "$CKPT_VAL90" "$SUB_VAL90" "$LOG_SUB2"

echo "[chain $(date -Is)] all phases complete"
