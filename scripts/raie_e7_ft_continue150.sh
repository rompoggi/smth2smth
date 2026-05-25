#!/usr/bin/env bash
# Continue E7 champion fine-tune to 150 epochs (bs=16, scaled LR, extended warmup).
#
# Prerequisite: checkpoints/track_a/ssl/e7_ft.last.pt (from champion FT).
#
# Usage (from repo root):
#   # After the 60-epoch champion job finishes (recommended):
#   bash scripts/raie_e7_ft_continue150.sh
#
#   # Or stop the running champion job and continue immediately from last.pt:
#   STOP_CURRENT_FT=1 bash scripts/raie_e7_ft_continue150.sh
#
# Monitor:
#   tail -f logs/raie_e7_ft_continue150_*.log
#   kill -0 "$(cat logs/raie_e7_ft_continue150_*.pid)"

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1 PYTHONPATH=src

PY="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing .venv — run: uv sync" >&2
  exit 1
fi

# Best champion weights (never use e7_ft.last.pt after a botched continue).
RESUME_CKPT="${RESUME_CKPT:-${REPO_ROOT}/checkpoints/track_a/ssl/e7_ft.pt}"
if [[ ! -s "$RESUME_CKPT" ]]; then
  echo "Missing resume checkpoint: $RESUME_CKPT" >&2
  echo "Run champion FT first (track_a_ssl_finetune_e7) until e7_ft.pt exists." >&2
  exit 1
fi

BATCH_TAG="${BATCH_TAG:-$(date +%Y%m%d)}"
LOG="logs/raie_e7_ft_continue150_${BATCH_TAG}.log"
PID="logs/raie_e7_ft_continue150_${BATCH_TAG}.pid"

if [[ "${STOP_CURRENT_FT:-0}" == "1" ]]; then
  for pat in "experiment=track_a_ssl_finetune_e7 track=a" "raie_e7_ft_champion"; do
    mapfile -t _pids < <(pgrep -f "$pat" || true)
    for p in "${_pids[@]}"; do
      if [[ "$p" != "$$" ]] && kill -0 "$p" 2>/dev/null; then
        echo "[continue150] stopping pid=$p ($pat)"
        kill "$p" || true
      fi
    done
  done
  sleep 3
fi

if [[ -f "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  echo "Continue-150 job already running pid=$(cat "$PID") log=$LOG" >&2
  exit 1
fi

echo "[continue150] $(date -Is) launching FT to 150 ep (bs=16, lr=5e-4 restored, epochs=150)"
echo "[continue150] resume_from=$RESUME_CKPT"

nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$PY" -u -m smth2smth.pipelines.train \
  experiment=track_a_ssl_finetune_e7_continue150 track=a \
  > "$LOG" 2>&1 &
echo $! > "$PID"
echo "Started pid=$(cat "$PID") log=$LOG"
echo "Tail: tail -f $LOG"
