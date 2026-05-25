#!/usr/bin/env bash
# Kaggle CSV for mae450-ft-f4 with champion TTA (see experiments/results_tta_ensembling.md).
#
# Champion: 3 patch-safe scales [0.857, 1.0, 1.143] + horizontal flip (6 views).
# Explicit test.num_segment=1 / num_crop=1 avoids the official 2×3 dense path from
# track_a_videomae_official_ssv2_ft (that preset defaults to test: vjepa2_official_2x3).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BATCH_TAG="${BATCH_TAG:-$(date +%Y%m%d)}"
CKPT="${REPO_ROOT}/checkpoints/track_a/videomaev2+ft/mae450-ft-f4.pt"
OUT="${REPO_ROOT}/submissions/mae450-ft-f4_tta_champion.csv"
LOG="${REPO_ROOT}/logs/mae450-ft-f4_submit_tta_champion_${BATCH_TAG}.log"
PIDFILE="${REPO_ROOT}/logs/mae450-ft-f4_submit_tta_champion.pid"

if [[ ! -s "$CKPT" ]]; then
  echo "Missing checkpoint: $CKPT" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1 PYTHONPATH=src
mkdir -p logs submissions

echo "[submit $(date -Is)] ckpt=$CKPT -> $OUT (champion TTA)" | tee "$LOG"
nohup uv run python -u -m smth2smth.pipelines.submit \
  track=a \
  training.checkpoint_path="$CKPT" \
  dataset.submission_output="$OUT" \
  training.tta=true \
  training.tta_flip=true \
  'training.tta_scales=[0.857,1.0,1.143]' \
  test.num_segment=1 \
  test.num_crop=1 \
  test.flip_tta=false \
  >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "PID=$(cat "$PIDFILE") log=$LOG"
