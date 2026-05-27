#!/usr/bin/env bash
# Wait for arch2-perceiver-q16-stab-s42 training, then Kaggle submit (champion TTA, single model).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_NAME="${RUN_NAME:-arch2-perceiver-q16-stab-s42}"
TAG="${TAG:-$(date +%Y%m%d)}"
TRAIN_PID_FILE="${TRAIN_PID_FILE:-logs/track_a/${RUN_NAME}_20260525.pid}"
CHAIN_LOG="${CHAIN_LOG:-logs/track_a/${RUN_NAME}-wait-submit-champion_${TAG}.log}"
SUBMIT_LOG="${SUBMIT_LOG:-logs/track_a/${RUN_NAME}-submit-final-champion-tta_${TAG}.log}"
CKPT="${CKPT:-checkpoints/track_a/videomaev2+ft/${RUN_NAME}.pt}"
SUBMIT_OUT="${SUBMIT_OUT:-submissions/track_a_${RUN_NAME}_final_champion_tta_${TAG}.csv}"
UV="${UV:-$(command -v uv)}"

log() { echo "[arch2-submit-chain $(date -Is)] $*" | tee -a "$CHAIN_LOG"; }

if [[ -z "${WAIT_PID:-}" ]]; then
  if [[ ! -f "$TRAIN_PID_FILE" ]]; then
    log "ERROR: missing $TRAIN_PID_FILE (set WAIT_PID=...)"
    exit 1
  fi
  WAIT_PID="$(cat "$TRAIN_PID_FILE")"
fi

log "Waiting for training pid=$WAIT_PID ($RUN_NAME)"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 30
done
log "Training pid $WAIT_PID exited."

if [[ ! -s "$CKPT" ]]; then
  log "ERROR: missing checkpoint $CKPT"
  exit 1
fi

log "Starting champion TTA submit: ckpt=$CKPT -> $SUBMIT_OUT"
{
  echo "# run: ${RUN_NAME}-submit-final-champion"
  echo "# started: $(date -Is)"
  echo "# track: a"
  echo "# experiment_doc: experiments/results_tta_ensembling.md"
  echo "# hydra: experiment=track_a_diverse_arch2_perceiver_stab_s42 training.tta=true champion scales3+flip"
  echo "# ckpt: ${CKPT}"
} >>"$SUBMIT_LOG"

export PYTHONPATH=src PYTHONUNBUFFERED=1 WANDB_MODE=disabled
"$UV" run python -u -m smth2smth.pipelines.submit \
  track=a \
  experiment=track_a_diverse_arch2_perceiver_stab_s42 \
  training.checkpoint_path="${CKPT}" \
  training.tta=true \
  training.tta_flip=true \
  'training.tta_scales=[0.857,1.0,1.143]' \
  test.num_segment=1 \
  test.num_crop=1 \
  test.flip_tta=false \
  training.batch_size=4 \
  training.wandb.enabled=false \
  dataset.submission_output="${SUBMIT_OUT}" \
  2>&1 | tee -a "$SUBMIT_LOG"

log "Submit finished -> $SUBMIT_OUT"
