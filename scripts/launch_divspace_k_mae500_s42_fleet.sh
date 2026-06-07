#!/usr/bin/env bash
# DivSpaceTime K sweep @ MAE500 SSL, seed 42 (train-only honest val).
# Usage: bash scripts/launch_divspace_k_mae500_s42_fleet.sh [sync|launch|health]
set -euo pipefail

REPO="${REPO:-/Data/romain.poggi/smth2smth}"
TAG="${TAG:-$(date +%Y%m%d)}"
MODE="${1:-launch}"
CKPT_DIR="${REPO}/checkpoints/track_a/divspace_s42"
SSL="${REPO}/checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep500.pt"
EXP=track_a_diverse_arch3_divided_st_stab

# host:run:K:wandb_run_id:resume_global_step (empty = fresh)
ENTRIES=(
  "brochet|DivSpaceTimeK6-mae500-s42|6|s42-replay-v2-DivSpaceTimeK6-mae500|94250"
  "lieu|DivSpaceTimeK9-mae500-s42|9|s42-replay-v2-DivSpaceTimeK9-mae500|123950"
  "murene|DivSpaceTimeK1-mae500-s42|1|NONE|NONE"
  "thon|DivSpaceTimeK12-mae500-s42|12|NONE|NONE"
)

sync_code() {
  local H="$1"
  local here
  here="$(hostname -s 2>/dev/null || hostname)"
  [[ "$H" == "$here" ]] && return 0
  rsync -az "${REPO}/src/" "${H}:${REPO}/src/"
  rsync -az "${REPO}/configs/" "${H}:${REPO}/configs/"
}

launch_one() {
  local H="$1" RUN="$2" K="$3" WID="$4" GSTEP="$5"
  echo "=== LAUNCH $H $RUN K=$K resume_step=${GSTEP:-fresh} ==="
  [[ "$WID" == NONE ]] && WID=""
  [[ "$GSTEP" == NONE ]] && GSTEP=""
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$RUN" "$K" "$WID" "$GSTEP" "$EXP" "$SSL" "$CKPT_DIR" "$TAG" <<'REMOTE'
set -eo pipefail
REPO="$1" RUN="$2" K="$3" WID="$4" GSTEP="$5" EXP="$6" SSL="$7" CKPT_DIR="$8" TAG="$9"
cd "$REPO"
mkdir -p logs/track_a/divspace_s42 "$CKPT_DIR"
LOG="logs/track_a/divspace_s42/${RUN}_${TAG}.log"

pkill -f "training.wandb.name=${RUN}" 2>/dev/null || true
sleep 2

RESUME_ARGS=()
if [[ -f "${CKPT_DIR}/${RUN}.last.pt" ]]; then
  RESUME_ARGS+=( "training.resume_from=${CKPT_DIR}/${RUN}.last.pt" )
elif [[ -f "${CKPT_DIR}/${RUN}.pt" ]]; then
  RESUME_ARGS+=( "training.resume_from=${CKPT_DIR}/${RUN}.pt" )
fi
if [[ -n "$GSTEP" ]]; then
  RESUME_ARGS+=( "training.resume_global_step=${GSTEP}" )
fi

{
  echo "# run: ${RUN}"
  echo "# started: $(date -Is)"
  echo "# track: a"
  echo "# experiment_doc: experiments/diverse_classifier_heads_post_mae.md"
  echo "# hydra: experiment=${EXP} model.temporal_layers=${K} seed=42 train-only"
  echo "# host: $(hostname)"
  echo "# ssl: ${SSL}"
  if [[ -n "$WID" ]]; then
    echo "# wandb_run_id: ${WID}"
    echo "# wandb_resume: allow"
  fi
} >>"$LOG"

WANDB_ENV=(WANDB_MODE=online)
if [[ -n "$WID" ]]; then
  WANDB_ENV+=(WANDB_RESUME=allow "WANDB_RUN_ID=${WID}")
fi

nohup env PYTHONPATH=src PYTHONUNBUFFERED=1 "${WANDB_ENV[@]}" \
  .venv/bin/python -u -m smth2smth.pipelines.train \
  track=a "experiment=${EXP}" seed=42 \
  "model.init_from=${SSL}" "model.temporal_layers=${K}" model.tube_t=1 dataset.num_frames=4 \
  dataset.include_val_in_train=false dataset.official_val_holdout_ratio=0 \
  "training.checkpoint_path=${CKPT_DIR}/${RUN}.pt" \
  "${RESUME_ARGS[@]}" \
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

health_one() {
  local H="$1" RUN="$2"
  local LOG="logs/track_a/divspace_s42/${RUN}_${TAG}.log"
  echo "=== HEALTH $H $RUN ==="
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$LOG" <<'HEALTH'
set -euo pipefail
REPO="$1" LOG="$2"
cd "$REPO"
test -f "$LOG"
grep -q Traceback "$LOG" && exit 1
grep -q '\[init_from\] loaded\|Resuming from checkpoint' "$LOG" || exit 1
grep -q '\[wandb\] run started:' "$LOG" || exit 1
grep -qE '\[.*\] step [0-9]+/' "$LOG" || exit 1
tail -6 "$LOG"
echo OK
HEALTH
}

case "$MODE" in
  sync)
    for e in "${ENTRIES[@]}"; do
      IFS='|' read -r H _ _ _ _ <<<"$e"
      sync_code "$H"
    done
    ;;
  launch)
    for e in "${ENTRIES[@]}"; do
      IFS='|' read -r H RUN K WID GSTEP <<<"$e"
      launch_one "$H" "$RUN" "$K" "$WID" "$GSTEP"
    done
    echo "Waiting 90s..."
    sleep 90
    ;;
  health)
    ok=0 fail=0
    for e in "${ENTRIES[@]}"; do
      IFS='|' read -r H RUN _ _ _ <<<"$e"
      if health_one "$H" "$RUN"; then ok=$((ok+1)); else fail=$((fail+1)); fi
    done
    echo "health: ok=$ok fail=$fail"
    ;;
  *)
    echo "Usage: $0 [sync|launch|health]"; exit 1 ;;
esac
