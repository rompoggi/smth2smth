#!/usr/bin/env bash
# Continue E5 stage-1 FT from epoch 50 → 150 after the initial 50-ep run ends.
#
# Does NOT touch the running stage-1 job. Launch manually once:
#   - logs/thon_e5_ft_stage1_rep50_20260516.log shows epoch 50/50 + process exit, AND
#   - test -s checkpoints/track_a/ssl/e5_ft.last.pt
#
# Usage (from repo root):
#   ./scripts/launch_thon_e5_ft_continue150.sh              # recommended: rewarm + new cosine
#   ./scripts/launch_thon_e5_ft_continue150.sh extend       # alternate: min-LR tail only

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export BATCH_TAG="${BATCH_TAG:-20260516}"

PY="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing .venv — run: uv sync" >&2
  exit 1
fi

MODE="${1:-rewarm}"
case "$MODE" in
  rewarm)
    EXPT="track_a_ssl_finetune_e5_continue150"
    LOG="logs/thon_e5_ft_continue150_rewarm_${BATCH_TAG}.log"
    PID="logs/thon_e5_ft_continue150_rewarm_${BATCH_TAG}.pid"
    ;;
  bs16)
    EXPT="track_a_ssl_finetune_e5_continue150_bs16"
    LOG="logs/thon_e5_ft_continue150_bs16_${BATCH_TAG}.log"
    PID="logs/thon_e5_ft_continue150_bs16_${BATCH_TAG}.pid"
    ;;
  extend)
    EXPT="track_a_ssl_finetune_e5_continue150_extend"
    LOG="logs/thon_e5_ft_continue150_extend_${BATCH_TAG}.log"
    PID="logs/thon_e5_ft_continue150_extend_${BATCH_TAG}.pid"
    ;;
  *)
    echo "Usage: $0 [rewarm|bs16|extend]" >&2
    exit 1
    ;;
esac

RESUME_CKPT="checkpoints/track_a/ssl/e5_ft.last.pt"
if [[ ! -s "$RESUME_CKPT" ]]; then
  echo "Missing resume checkpoint: $RESUME_CKPT" >&2
  echo "Wait for stage-1 (50 ep) to finish and write e5_ft.last.pt." >&2
  exit 1
fi

if pgrep -f "smth2smth.pipelines.train experiment=track_a_ssl_finetune_e5 track=a" \
  >/dev/null 2>&1; then
  echo "[warn] A track_a_ssl_finetune_e5 job may still be running." >&2
  echo "       Do not launch continuation until stage-1 has exited." >&2
  exit 1
fi

mkdir -p logs checkpoints/track_a/ssl

nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$PY" -u -m smth2smth.pipelines.train \
  "experiment=${EXPT}" track=a \
  > "$LOG" 2>&1 &
echo $! > "$PID"
echo "Started E5 FT continuation (${MODE}) pid=$(cat "$PID") log=$LOG"
echo "Resume from: $RESUME_CKPT"
echo "Monitor: tail -f $LOG"
