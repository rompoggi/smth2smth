#!/usr/bin/env bash
# Quick health check for Track B r1 runs (SSH from any machine with fish access).
#   bash scripts/check_track_b_r1_health.sh
#   bash scripts/check_track_b_r1_health.sh --only saumon
set -euo pipefail

REPO=/Data/romain.poggi/smth2smth
TAG=20260527
ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) ONLY="${2:?}"; shift 2 ;;
        *) echo "Unknown: $1" >&2; exit 1 ;;
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

OK=0; DEAD=0; ERR=0

for i in "${!HOSTS[@]}"; do
    H="${HOSTS[$i]}"; R="${RUNS[$i]}"
    if [[ -n "$ONLY" && "$ONLY" != "$H" && "$ONLY" != "$R" ]]; then
        continue
    fi
    LOG="${REPO}/logs/track_b/${R}_${TAG}.log"
    RESULT=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$H" bash -s -- "$REPO" "$R" "$LOG" <<'REMOTE'
REPO="$1"; R="$2"; LOG="$3"
cd "$REPO"
PROC=$(pgrep -cf "pipelines.train.*track_b_${R}" 2>/dev/null || echo 0)
STEP=$(grep -E '\] step [0-9]+/' "$LOG" 2>/dev/null | tail -1 | sed 's/^ *//' || true)
CRASH=$(grep -cE 'Traceback|CUDA error' "$LOG" 2>/dev/null || echo 0)
WB=$(grep -oP '\[wandb\] run started: \Khttps://[^\s]+' "$LOG" 2>/dev/null | tail -1 || true)
echo "proc=${PROC} crash=${CRASH} wb=${WB:-none} step=${STEP:-NO_STEP}"
REMOTE
    2>/dev/null || echo "proc=? crash=? wb=none step=SSH_FAIL")

    PROC=$(echo "$RESULT" | grep -oP 'proc=\K[0-9?]+')
    STEP=$(echo "$RESULT" | sed 's/.*step=//')
    CRASH=$(echo "$RESULT" | grep -oP 'crash=\K[0-9]+')
    WB=$(echo "$RESULT" | grep -oP 'wb=\K\S+')

    if [[ "$STEP" == "SSH_FAIL" ]]; then
        STATUS="SSH_FAIL"
        ((ERR++)) || true
    elif [[ "${CRASH:-0}" -gt 0 ]]; then
        STATUS="CRASHED(${CRASH})"
        ((ERR++)) || true
    elif echo "$STEP" | grep -qE 'step [0-9]+/'; then
        STATUS="OK"
        ((OK++)) || true
    else
        STATUS="NO_STEP(proc=${PROC})"
        ((DEAD++)) || true
    fi

    printf "%-12s %-28s %-10s  %s\n" "$H" "$R" "$STATUS" "${STEP:-}"
    [[ "$WB" != "none" ]] && printf "%-12s %-28s %s\n" "" "" "  W&B: $WB"
done

echo ""
echo "Summary: OK=${OK}  NO_STEP/DEAD=${DEAD}  CRASH/ERR=${ERR}"
