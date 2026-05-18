#!/bin/bash
# Wait for val_plus2 classic submit, then run val_plus3.
set -euo pipefail
cd "$(dirname "$0")/.."
PID=$(cat logs/track_b_vitl_fulltrain_lowlr_val_plus2_submit.pid)
echo "Waiting for plus2 submit PID ${PID}..."
while kill -0 "${PID}" 2>/dev/null; do sleep 20; done
echo "plus2 done at $(date)"
LOG2="logs/track_b_vitl_fulltrain_lowlr_val_plus3_submit.log"
PYTHONUNBUFFERED=1 nohup env PYTHONPATH=src .venv/bin/python -u -m smth2smth.pipelines.submit \
  track=b experiment=track_b_vjepa2_vitl_fulltrain_lowlr \
  training.checkpoint_path=checkpoints/track_b/vitl_fulltrain_lora_lowlr.pt \
  dataset.submission_output=submissions/track_b_vitl_fulltrain_lora_lowlr_val_plus3.csv \
  > "${LOG2}" 2>&1 < /dev/null &
echo $! > logs/track_b_vitl_fulltrain_lowlr_val_plus3_submit.pid
echo "Started plus3 PID=$(cat logs/track_b_vitl_fulltrain_lowlr_val_plus3_submit.pid)"
