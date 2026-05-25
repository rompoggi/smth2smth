#!/usr/bin/env bash
# Queue: wait for ablette mHC s42 honest FT -> submit (best TTA / no TTA) ->
# val90 holdout FT -> submit (best TTA / no TTA).
#
# Holdout TTA sweep (ablette mHC s42): pick arms from val top-1 on 10% holdout.
#   Best TTA:  3 scales + flip [0.857, 1.0, 1.143] -> 0.5414 (+1.92 pp vs no TTA)
#   No TTA:    single scale [1.0], no flip          -> 0.5222 (baseline)
#   Skip:      3 scales, no flip (0.5222, no top-1 gain vs baseline)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH=src

PY="${REPO_ROOT}/.venv/bin/python"
BATCH_TAG="${BATCH_TAG:-20260519}"
SEED=42

TRAIN_PID_FILE="logs/hc/ablette_hc_mhc_s42_20260518.pid"
TRAIN_LOG="logs/hc/ablette_hc_mhc_s42_20260518.log"
PIPELINE_LOG="logs/hc/ablette_hc_mhc_s42_pipeline_${BATCH_TAG}.log"
PIPELINE_PID_FILE="logs/hc/ablette_hc_mhc_s42_pipeline_${BATCH_TAG}.pid"

CKPT_FT="${REPO_ROOT}/checkpoints/track_a/hc/mhc_n4_sk3_s${SEED}_ft.pt"
CKPT_HOLDOUT="${REPO_ROOT}/checkpoints/track_a/hc/mhc_n4_sk3_s${SEED}_ft_val90_holdout.pt"

# Best holdout TTA: 3 ViT scales + horizontal flip (tied with 0.875/1.0/1.125; use project default sides).
SUBMIT_TTA="submissions/track_a_hc_mhc_n4_sk3_s${SEED}_tta.csv"
# True no-TTA baseline (not "scales without flip" — that matched no-TTA on holdout top-1).
SUBMIT_NO_TTA="submissions/track_a_hc_mhc_n4_sk3_s${SEED}_no_tta.csv"
SUBMIT_HOLDOUT_TTA="submissions/track_a_hc_mhc_n4_sk3_s${SEED}_val90_tta.csv"
SUBMIT_HOLDOUT_NO_TTA="submissions/track_a_hc_mhc_n4_sk3_s${SEED}_val90_no_tta.csv"

TTA_SCALES='[0.857,1.0,1.143]'

log() {
  echo "[pipeline $(date -Is)] $*"
}

wait_for_train() {
  log "Waiting for honest FT to finish (pid file: ${TRAIN_PID_FILE})"
  if [[ -f "$TRAIN_PID_FILE" ]]; then
    local pid
    pid="$(cat "$TRAIN_PID_FILE")"
    while kill -0 "$pid" 2>/dev/null; do
      sleep 30
    done
    log "PID ${pid} from pid file exited."
  fi
  while pgrep -f "smth2smth.pipelines.train experiment=track_a_hc_ablation_mhc track=a seed=${SEED}" >/dev/null 2>&1; do
    log "Still waiting on track_a_hc_ablation_mhc train (pgrep)..."
    sleep 30
  done
  if [[ ! -s "$CKPT_FT" ]]; then
    log "ERROR: missing best checkpoint: $CKPT_FT"
    exit 1
  fi
  if ! grep -q "Done\\. Best val top1" "$TRAIN_LOG" 2>/dev/null; then
    log "WARN: train log has no 'Done.' line yet; proceeding because ckpt exists."
  else
    log "Train log shows completion."
  fi
  log "Honest FT done. Best ckpt: $CKPT_FT"
}

run_submit_best_tta() {
  local ckpt="$1"
  local out="$2"
  local tag="$3"
  local slog="logs/hc/ablette_hc_mhc_s42_submit_${tag}_tta_${BATCH_TAG}.log"
  log "Submit (${tag}, best TTA): 3 scales + flip ${TTA_SCALES} -> $out"
  env PYTHONUNBUFFERED=1 PYTHONPATH=src \
    "$PY" -u -m smth2smth.pipelines.submit \
    track=a seed="${SEED}" \
    training.checkpoint_path="$ckpt" \
    training.tta=true \
    training.tta_flip=true \
    "training.tta_scales=${TTA_SCALES}" \
    dataset.submission_output="$out" \
    2>&1 | tee -a "$slog"
  log "Wrote $out (log: $slog)"
}

run_submit_no_tta() {
  local ckpt="$1"
  local out="$2"
  local tag="$3"
  local slog="logs/hc/ablette_hc_mhc_s42_submit_${tag}_no_tta_${BATCH_TAG}.log"
  log "Submit (${tag}, no TTA): single view scale=1.0 -> $out"
  env PYTHONUNBUFFERED=1 PYTHONPATH=src \
    "$PY" -u -m smth2smth.pipelines.submit \
    track=a seed="${SEED}" \
    training.checkpoint_path="$ckpt" \
    training.tta=false \
    dataset.submission_output="$out" \
    2>&1 | tee -a "$slog"
  log "Wrote $out (log: $slog)"
}

run_holdout_ft() {
  local slog="logs/hc/ablette_hc_mhc_s42_val90_holdout_${BATCH_TAG}.log"
  log "Starting val90 holdout FT -> $CKPT_HOLDOUT (log: $slog)"
  env PYTHONUNBUFFERED=1 PYTHONPATH=src \
    "$PY" -u -m smth2smth.pipelines.train \
    experiment=track_a_hc_ablation_mhc_val90_holdout track=a seed="${SEED}" \
    2>&1 | tee -a "$slog"
  local ec=${PIPESTATUS[0]}
  if [[ "$ec" -ne 0 ]]; then
    log "ERROR: holdout FT exited with code $ec"
    exit "$ec"
  fi
  if [[ ! -s "$CKPT_HOLDOUT" ]]; then
    log "ERROR: holdout checkpoint missing: $CKPT_HOLDOUT"
    exit 1
  fi
  log "Holdout FT done. Best ckpt: $CKPT_HOLDOUT"
}

mkdir -p logs/hc submissions checkpoints/track_a/hc

{
  log "========== ablette mHC s42 post-train pipeline start =========="
  log "pipeline pid=$$ (written to ${PIPELINE_PID_FILE})"
  echo "$$" > "$PIPELINE_PID_FILE"

  wait_for_train

  run_submit_best_tta "$CKPT_FT" "$SUBMIT_TTA" "ft"
  run_submit_no_tta "$CKPT_FT" "$SUBMIT_NO_TTA" "ft"

  run_holdout_ft

  run_submit_best_tta "$CKPT_HOLDOUT" "$SUBMIT_HOLDOUT_TTA" "val90"
  run_submit_no_tta "$CKPT_HOLDOUT" "$SUBMIT_HOLDOUT_NO_TTA" "val90"

  log "========== pipeline complete =========="
} 2>&1 | tee -a "$PIPELINE_LOG"
