#!/usr/bin/env bash
# Idempotent auto-resume for a Round-3 diverse-heads run after the ~06:30 admin
# cull (or any crash). Safe to run from cron repeatedly: it is a NO-OP when the
# run is still alive or has legitimately finished; otherwise it resumes from
# <run>.last.pt into the SAME log file and the SAME W&B run.
#
# Usage: auto_resume_round3.sh RUN_NAME EXPERIMENT "EXTRA overrides"
#   RUN_NAME    e.g. arch4-aim-trainonly
#   EXPERIMENT  e.g. track_a_diverse_arch4_aim
#   EXTRA       the exact CLI overrides used at first launch (Q/K/holdout/++ keys)
#
# Pair with a per-host crontab line guarded by flock, e.g.:
#   */15 * * * * flock -n /tmp/resume_<run>.lock \
#     /Data/romain.poggi/smth2smth/scripts/auto_resume_round3.sh <run> <exp> "<extra>" \
#     >> /Data/romain.poggi/smth2smth/logs/track_a/<run>.autoresume.log 2>&1
set -uo pipefail

RUN_NAME="${1:?run name}"; EXPERIMENT="${2:?experiment}"; EXTRA="${3:-}"
REPO=/Data/romain.poggi/smth2smth
cd "$REPO" || exit 9

PY="$REPO/.venv/bin/python"
SSL="$REPO/checkpoints/track_a/ssl/videomaev2_t4native_encoder_ep500.pt"
CKPT="$REPO/checkpoints/track_a/videomaev2+ft/${RUN_NAME}.pt"
LAST="$REPO/checkpoints/track_a/videomaev2+ft/${RUN_NAME}.last.pt"
PIDF="$REPO/logs/track_a/${RUN_NAME}.autoresume.pid"
# Append to the newest existing log for this run (distributed-run-log: same log on resume).
LOG=$(ls -1t "$REPO"/logs/track_a/${RUN_NAME}_*.log 2>/dev/null | head -1)
[ -z "${LOG:-}" ] && LOG="$REPO/logs/track_a/${RUN_NAME}_$(date +%Y%m%d).log"

ts(){ date -Is; }

# 1) Already running? -> nothing to do.
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null; then exit 0; fi
if pgrep -f "training.wandb.name=${RUN_NAME} " >/dev/null 2>&1; then exit 0; fi

# 2) Legitimately finished (normal completion or early stop)? -> do not relaunch.
if [ -f "$LOG" ] && grep -qE "Early stopping triggered:|^Done\. Best val " "$LOG"; then
  exit 0
fi

# 3) Down and unfinished -> resume.
test -f "$SSL" || { echo "$(ts) ABORT ${RUN_NAME}: missing SSL $SSL"; exit 1; }
RESUME=""
[ -f "$LAST" ] && RESUME="training.resume_from=${LAST}"
WID=$(grep -oE "runs/[A-Za-z0-9]+" "$LOG" 2>/dev/null | tail -1 | cut -d/ -f2)

export PYTHONPATH=src PYTHONUNBUFFERED=1
if [ -n "${WID:-}" ]; then export WANDB_RESUME=allow WANDB_RUN_ID="$WID"; fi

echo "# === AUTO-RESUME $(ts) run=${RUN_NAME} resume_from=${LAST##*/} wandb_id=${WID:-none} ===" >> "$LOG"
nohup "$PY" -u -m smth2smth.pipelines.train \
  experiment="${EXPERIMENT}" \
  model.init_from="${SSL}" \
  training.checkpoint_path="${CKPT}" \
  training.wandb.name="${RUN_NAME}" \
  training.wandb.project=smth2smth-diverse-heads \
  ${EXTRA} ${RESUME} >> "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDF"
echo "# auto-resume launched pid $(cat "$PIDF") $(ts)" >> "$LOG"
