#!/usr/bin/env bash
# Resume Track B r1 fleet after VM reboot (SSH launcher — run from any machine
# that can ``ssh anguille``, ``ssh saumon``, etc. without a password prompt).
#
# Full fleet:
#   bash scripts/resume_track_b_r1_after_reboot.sh
#
# Single host or run (dry-run / test):
#   bash scripts/resume_track_b_r1_after_reboot.sh --only saumon
#   bash scripts/resume_track_b_r1_after_reboot.sh --only r1_b11_4block_lr3e4
#
# Health check after ~90s:
#   bash scripts/check_track_b_r1_health.sh --only saumon
#
# Per host: skip if already running; append ``# resumed:`` to today's log;
# resume from ``*.last.pt`` (end-of-epoch) else ``*.pt`` else fresh start;
# WANDB_RESUME when a run URL is in the log.
set -euo pipefail

REPO=/Data/romain.poggi/smth2smth
TAG=20260527
ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)
            ONLY="${2:?--only requires host or run name}"
            shift 2
            ;;
        -h|--help)
            sed -n '2,18p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown arg: $1 (try --only saumon)" >&2
            exit 1
            ;;
    esac
done

HOSTS=(anguille barbeau  barbue    baudroie  gardon
       lieu     murene   piranha   raie      roussette
       saumon   truite)
RUNS=( r1_b01_hf_lr5e3
       r1_b02_hf_lr3e3
       r1_b03_hf_lr1e3
       r1_b04_hf_lr3e4
       r1_b05_hf_lr1e4
       r1_b06_hf_wd001
       r1_b07_mq16_lr1e3
       r1_b08_mq16_lr3e3
       r1_b09_mq8_lr1e3
       r1_b10_mq32_lr1e3
       r1_b11_4block_lr3e4
       r1_b12_hf_dora_mlp)

resume_one() {
    local H="$1"
    local R="$2"
    local E="track_b_${R}"
    local LOG="logs/track_b/${R}_${TAG}.log"
    local CKPT="checkpoints/track_b/${R}.pt"
    local LAST="checkpoints/track_b/${R}.last.pt"

    echo "=== RESUME $H -> $R ==="

    ssh -o BatchMode=yes -o ConnectTimeout=15 "$H" bash -s -- \
        "$REPO" "$R" "$E" "$LOG" "$CKPT" "$LAST" <<'REMOTE'
set -euo pipefail
REPO="$1"; R="$2"; E="$3"; LOG="$4"; CKPT="$5"; LAST="$6"
cd "$REPO"

# ── already running? ──────────────────────────────────────────────────────────
if pgrep -af "experiment=${E}" >/dev/null 2>&1; then
    echo "SKIP: ${E} already running on $(hostname)"
    exit 0
fi

mkdir -p logs/track_b checkpoints/track_b

# ── choose checkpoint ─────────────────────────────────────────────────────────
RESUME_FLAG=""
if [[ -f "$LAST" ]]; then
    RESUME_FLAG="training.resume_from=$LAST"
    echo "  checkpoint: $LAST (last)"
elif [[ -f "$CKPT" ]]; then
    RESUME_FLAG="training.resume_from=$CKPT"
    echo "  checkpoint: $CKPT (best)"
else
    echo "  checkpoint: none — starting fresh"
fi

# ── extract W&B run id from existing log ──────────────────────────────────────
WB_ENV=""
if [[ -f "$LOG" ]]; then
    WB_URL=$(grep -oP '\[wandb\] run started: \Khttps://[^\s]+' "$LOG" | tail -1 || true)
    if [[ -n "$WB_URL" ]]; then
        WB_ID=$(echo "$WB_URL" | grep -oP 'runs/\K[^/?]+' || true)
        if [[ -n "$WB_ID" ]]; then
            WB_ENV="WANDB_RESUME=allow WANDB_RUN_ID=${WB_ID}"
            echo "  W&B resume: $WB_ID"
        fi
    fi
fi

# ── append resume header to log ───────────────────────────────────────────────
{
    echo ""
    echo "# resumed: $(date -Is)"
    echo "# host: $(hostname)"
    [[ -n "$RESUME_FLAG" ]] && echo "# resume_ckpt: ${RESUME_FLAG#training.resume_from=}"
    [[ -n "$WB_ENV" ]]      && echo "# wandb_resume: ${WB_ENV}"
} >>"$LOG"

# ── launch ────────────────────────────────────────────────────────────────────
nohup env PYTHONPATH=src PYTHONUNBUFFERED=1 $WB_ENV \
    .venv/bin/python -u -m smth2smth.pipelines.train \
    track=b \
    "experiment=${E}" \
    "training.checkpoint_path=${CKPT}" \
    ${RESUME_FLAG} \
    >>"$LOG" 2>&1 &
PID=$!
echo "pid=${PID} log=${LOG}"
disown "$PID"
REMOTE
}

# ── iterate hosts ─────────────────────────────────────────────────────────────
FAILED=()
for i in "${!HOSTS[@]}"; do
    H="${HOSTS[$i]}"
    R="${RUNS[$i]}"
    if [[ -n "$ONLY" && "$ONLY" != "$H" && "$ONLY" != "$R" ]]; then
        continue
    fi
    if resume_one "$H" "$R"; then
        :
    else
        echo "ERROR: $H / $R failed to launch"
        FAILED+=("$H/$R")
    fi
done

echo ""
echo "=== RESUME COMPLETE ==="
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed hosts: ${FAILED[*]}"
else
    echo "All 12 hosts contacted successfully."
fi
echo ""
echo "Health check in 90s:"
echo "  bash scripts/check_track_b_r1_health.sh"
