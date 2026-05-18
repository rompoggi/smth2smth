#!/usr/bin/env bash
# Resume E4 trainval FT after a pause (submit window). Appends to the same train log.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BATCH_TAG="${BATCH_TAG:-20260516}"
PY="${REPO_ROOT}/.venv/bin/python"
LOG="logs/truite_e4_trainval_ft_${BATCH_TAG}.log"
PID="logs/truite_e4_trainval_ft_${BATCH_TAG}.pid"
LAST="${REPO_ROOT}/checkpoints/track_a/e4_ft_trainval.last.pt"

if [[ ! -x "$PY" ]]; then
  echo "Missing .venv — run: uv sync" >&2
  exit 1
fi
if [[ ! -s "$LAST" ]]; then
  echo "Missing resume checkpoint: $LAST" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1 PYTHONPATH=src

{
  echo ""
  echo "========== resume after submit pause $(date -Is) =========="
} >> "$LOG"

nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$PY" -u -m smth2smth.pipelines.train \
  experiment=track_a_e4_tsm_trainval_ft track=a \
  training.resume_from="$LAST" \
  training.reset_epoch_on_resume=false \
  training.resume_apply_cfg_lr=false \
  >> "$LOG" 2>&1 &
echo $! > "$PID"
echo "Resumed trainval FT pid=$(cat "$PID") log=$LOG (from epoch in $LAST)"
