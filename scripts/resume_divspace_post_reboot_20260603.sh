#!/usr/bin/env bash
# Resume DivSpaceTime K runs interrupted by GPU restart (2026-06-03).
# Skips intentionally PAUSED *-NoStab Q16 and stopped K12.
# Usage: bash scripts/resume_divspace_post_reboot_20260603.sh [resume|chain-q16|health|all]
set -euo pipefail

REPO="${REPO:-/Data/romain.poggi/smth2smth}"
cd "$REPO"
TAG="${TAG:-$(date +%Y%m%d)}"
EXP=track_a_diverse_arch3_divided_st_stab
SSL="${REPO}/checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep500.pt"
MODE="${1:-all}"

# host:run:K:seed:wandb_id
ENTRIES=(
  ablette:DivSpaceTimeK1-mae500-s43:1:43:ntuwnv5r
  anguille:DivSpaceTimeK3-mae500-s43:3:43:i7ado1gd
  barbue:DivSpaceTimeK6-mae500-s43:6:43:txwwef5b
  carrelet:DivSpaceTimeK9-mae500-s43:9:43:0sguvl8z
  piranha:DivSpaceTimeK1-mae500-s44:1:44:9cdrb7z6
  raie:DivSpaceTimeK3-mae500-s44:3:44:z0zwj43l
  requin:DivSpaceTimeK6-mae500-s44:6:44:01gnvhqm
  roussette:DivSpaceTimeK9-mae500-s44:9:44:533jwygf
  brochet:DivSpaceTimeK6-mae500-s42:6:42:s42-replay-v2-DivSpaceTimeK6-mae500
  gardon:DivSpaceTimeK1-mae500-s42:1:42:pu325vfh
  lieu:DivSpaceTimeK9-mae500-s42:9:42:s42-replay-v2-DivSpaceTimeK9-mae500
)

ckpt_global_step() {
  local H="$1" CKPT="$2"
  ssh -o BatchMode=yes "$H" "$REPO/.venv/bin/python" - "$CKPT" <<'PY'
import sys, torch
p = sys.argv[1]
c = torch.load(p, map_location="cpu", weights_only=False)
gs = (c.get("extra") or {}).get("global_step")
print(int(gs) if gs is not None else 0)
PY
}

resume_divspace() {
  local H="$1" RUN="$2" K="$3" SEED="$4" WID="$5"
  local CKPT_DIR="${REPO}/checkpoints/track_a/divspace_s${SEED}"
  local LOG_DIR="logs/track_a/divspace_s${SEED}"
  local RESUME_PT="${CKPT_DIR}/${RUN}.last.pt"
  local GSTEP
  GSTEP=$(ckpt_global_step "$H" "$RESUME_PT") || {
    echo "FAIL: no ckpt $H $RUN"
    return 1
  }
  echo "=== RESUME $H $RUN K=$K ep~? gs=$GSTEP wandb=$WID ==="
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$RUN" "$K" "$SEED" "$WID" "$EXP" "$SSL" "$CKPT_DIR" "$LOG_DIR" "$RESUME_PT" "$GSTEP" <<'REMOTE'
set -euo pipefail
REPO="$1" RUN="$2" K="$3" SEED="$4" WID="$5" EXP="$6" SSL="$7" CKPT_DIR="$8" LOG_DIR="$9" RESUME_PT="${10}" GSTEP="${11}"
cd "$REPO"
mkdir -p "$LOG_DIR" "$CKPT_DIR"
LOG=$(ls -t "${LOG_DIR}/${RUN}"_*.log 2>/dev/null | head -1)
[[ -n "$LOG" ]] || LOG="${LOG_DIR}/${RUN}_${TAG:-unknown}.log"
test -f "$RESUME_PT" || { echo "ABORT: missing $RESUME_PT" >&2; exit 1; }
test -f "$SSL" || { echo "ABORT: missing SSL" >&2; exit 1; }
pkill -f "training.wandb.name=${RUN}" 2>/dev/null || true
sleep 2
if pgrep -af "training.wandb.name=${RUN}" | grep -q python; then
  echo "SKIP: already running"
  exit 0
fi
{
  echo ""
  echo "# === RESUME $(date -Is) post GPU restart ==="
  echo "# resume_from: ${RESUME_PT}"
  echo "# resume_global_step: ${GSTEP}"
  echo "# wandb_run_id: ${WID}"
} >>"$LOG"
nohup env PYTHONPATH=src PYTHONUNBUFFERED=1 WANDB_MODE=online \
  WANDB_RESUME=allow "WANDB_RUN_ID=${WID}" \
  .venv/bin/python -u -m smth2smth.pipelines.train \
  track=a "experiment=${EXP}" "seed=${SEED}" \
  "model.init_from=${SSL}" "model.temporal_layers=${K}" model.tube_t=1 dataset.num_frames=4 \
  dataset.include_val_in_train=false dataset.official_val_holdout_ratio=0 \
  "training.checkpoint_path=${CKPT_DIR}/${RUN}.pt" \
  "training.resume_from=${RESUME_PT}" \
  "training.resume_global_step=${GSTEP}" \
  "training.wandb.project=smth2smth-frame-ablation" \
  ++training.wandb.group=ft-mae-scaling \
  "training.wandb.name=${RUN}" \
  ++training.wandb.config.head_type="DivSpaceTimeK${K}" \
  ++training.wandb.config.temporal_layers="${K}" \
  ++training.wandb.config.pretrain_epochs=500 \
  >>"$LOG" 2>&1 &
echo "trainer_pid=$! log=$LOG"
REMOTE
}

do_chain_q16() {
  local CHAIN_LOG="logs/track_a/chain_q16_mae500_stab_${TAG}.log"
  mkdir -p logs/track_a
  {
    echo "# chain: Q16 mae500 stab after DivSpaceTime K9/K6 s42 (restarted post GPU reboot)"
    echo "# started: $(date -Is)"
  } >"$CHAIN_LOG"
  nohup env REPO="$REPO" TAG="$TAG" bash -s >>"$CHAIN_LOG" 2>&1 <<'CHAIN' &
set -euo pipefail
REPO="${REPO:-/Data/romain.poggi/smth2smth}"
TAG="${TAG:-$(date +%Y%m%d)}"
EXP=track_a_diverse_arch2_perceiver_stab
SSL="${REPO}/checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep500.pt"

wait_run() {
  local H="$1" RUN="$2"
  echo "[$(date -Is)] waiting ${RUN} on ${H}..."
  while ssh -o BatchMode=yes "$H" "pgrep -f 'training.wandb.name=${RUN}'" >/dev/null 2>&1; do
    sleep 120
  done
  echo "[$(date -Is)] ${RUN} finished on ${H}"
}

launch_q16_stab() {
  local H="$1" RUN="$2" SEED="$3"
  local LOG="logs/track_a/${RUN}_${TAG}.log"
  echo "[$(date -Is)] launch stab ${RUN} on ${H}"
  ssh -o BatchMode=yes "$H" "mkdir -p ${REPO}/checkpoints/track_a/ssl/pretrain"
  rsync -az "$SSL" "${H}:${SSL}" 2>/dev/null || true
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$RUN" "$SSL" "$SEED" "$LOG" "$EXP" <<'LAUNCH'
set -euo pipefail
REPO="$1" RUN="$2" SSL="$3" SEED="$4" LOG="$5" EXP="$6"
cd "$REPO"
mkdir -p logs/track_a checkpoints/track_a/videomaev2+ft
if pgrep -af "smth2smth.pipelines.train" | grep -v pgrep | grep -q .; then
  echo "ABORT: GPU busy on $(hostname -s)" >&2; exit 1
fi
: >"$LOG"
{
  echo "# run: ${RUN}"
  echo "# started: $(date -Is)"
  echo "# track: a"
  echo "# experiment_doc: experiments/diverse_classifier_heads_post_mae.md"
  echo "# hydra: experiment=${EXP} seed=${SEED} model.head_queries=16 T=4 train-only"
  echo "# recipe: stab"
} >>"$LOG"
nohup env PYTHONPATH=src PYTHONUNBUFFERED=1 WANDB_MODE=online \
  .venv/bin/python -u -m smth2smth.pipelines.train \
  track=a "experiment=${EXP}" "seed=${SEED}" \
  "model.init_from=${SSL}" model.head_queries=16 model.tube_t=1 dataset.num_frames=4 \
  dataset.include_val_in_train=false dataset.official_val_holdout_ratio=0 \
  "training.checkpoint_path=checkpoints/track_a/videomaev2+ft/${RUN}.pt" \
  "training.wandb.project=smth2smth-frame-ablation" \
  ++training.wandb.group=ft-mae-scaling \
  "training.wandb.name=${RUN}" \
  ++training.wandb.config.head_type=perceiverQ16 \
  ++training.wandb.config.pretrain_epochs=500 \
  ++training.wandb.config.num_queries=16 \
  ++training.wandb.config.recipe=stab \
  >>"$LOG" 2>&1 &
echo "trainer_pid=$! log=$LOG"
LAUNCH
}

# If already finished, launch immediately; else wait.
if ssh -o BatchMode=yes lieu "pgrep -f 'training.wandb.name=DivSpaceTimeK9-mae500-s42'" >/dev/null 2>&1; then
  wait_run lieu DivSpaceTimeK9-mae500-s42
else
  echo "[$(date -Is)] DivSpaceTimeK9-mae500-s42 not running on lieu (assume done or idle)"
fi
launch_q16_stab lieu perceiverQ16-mae500-s43 43

if ssh -o BatchMode=yes brochet "pgrep -f 'training.wandb.name=DivSpaceTimeK6-mae500-s42'" >/dev/null 2>&1; then
  wait_run brochet DivSpaceTimeK6-mae500-s42
else
  echo "[$(date -Is)] DivSpaceTimeK6-mae500-s42 not running on brochet"
fi
launch_q16_stab brochet perceiverQ16-mae500-s44 44
echo "[$(date -Is)] Q16 chain done"
CHAIN
  echo "chain_pid=$! log=$CHAIN_LOG"
}

do_health() {
  local ok=0 fail=0
  for e in "${ENTRIES[@]}"; do
    IFS=: read -r H RUN _K _S _W <<<"$e"
    echo "=== HEALTH $H $RUN ==="
    if ssh -o BatchMode=yes "$H" bash -s <<HEALTH
set -e
pgrep -af "training.wandb.name=${RUN}" | grep python | head -1
grep -qE '\\[.*\\] step [0-9]+/' \$(ls -t ${REPO}/logs/track_a/divspace_s*/${RUN}_*.log | head -1) && echo OK
HEALTH
    then ok=$((ok + 1)); else fail=$((fail + 1)); fi
  done
  echo "health: ok=$ok fail=$fail"
}

case "$MODE" in
  resume)
    for e in "${ENTRIES[@]}"; do
      IFS=: read -r H RUN K SEED WID <<<"$e"
      resume_divspace "$H" "$RUN" "$K" "$SEED" "$WID" || echo "FAIL $H $RUN"
    done
    ;;
  chain-q16) do_chain_q16 ;;
  health) do_health ;;
  all)
    for e in "${ENTRIES[@]}"; do
      IFS=: read -r H RUN K SEED WID <<<"$e"
      resume_divspace "$H" "$RUN" "$K" "$SEED" "$WID" || echo "FAIL $H $RUN"
    done
    do_chain_q16
    echo "Waiting 90s..."
    sleep 90
    do_health || true
    ;;
  *) echo "Usage: $0 [resume|chain-q16|health|all]"; exit 1 ;;
esac
echo "Done ($MODE)."
