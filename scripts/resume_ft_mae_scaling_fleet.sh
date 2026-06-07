#!/usr/bin/env bash
# Resume interrupted ft-mae-scaling / perceiver Q seed runs (same log + W&B run id).
# Usage: bash scripts/resume_ft_mae_scaling_fleet.sh [resume|health]
set -euo pipefail

REPO="${REPO:-/Data/romain.poggi/smth2smth}"
MODE="${1:-resume}"

# host:run:wandb_id:kind  (kind = meanpool | perceiver)
ENTRIES=(
  brochet:meanpool-mae050-s44:cnjvhknv:meanpool
  gardon:perceiverQ2-mae500-s43:7ew6eok8:perceiver
  lieu:perceiverQ2-mae500-s44:gd5o0lwj:perceiver
  lotte:perceiverQ4-mae500-s43:ldvozwwn:perceiver
  mulet:perceiverQ4-mae500-s44:3kmbiaki:perceiver
  murene:perceiverQ8-mae500-s43:wwvaib7c:perceiver
  piranha:perceiverQ8-mae500-s44:2wusde6o:perceiver
  raie:perceiverQ32-mae500-s43:20ykyagi:perceiver
  requin:perceiverQ32-mae500-s44:532pq1s0:perceiver
  roussette:perceiverQ64-mae500-s42:e4h78uqj:perceiver
  sole:perceiverQ64-mae500-s43:0xxh9rr3:perceiver
  thon:perceiverQ64-mae500-s44:e363jmlq:perceiver
)

parse_q() { echo "${1#perceiverQ}" | cut -d- -f1; }
parse_seed() { echo "${1##*-s}"; }

resume_one() {
  local H="$1" RUN="$2" WID="$3" KIND="$4"
  echo "=== RESUME $H $RUN wandb=$WID ==="
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$RUN" "$WID" "$KIND" <<'REMOTE'
set -euo pipefail
REPO="$1" RUN="$2" WID="$3" KIND="$4"
cd "$REPO"

LOG=$(ls -t "$REPO/logs/track_a/${RUN}"_*.log 2>/dev/null | head -1)
if [[ -z "$LOG" ]]; then
  echo "ABORT: no log for $RUN on $(hostname -s)" >&2
  exit 1
fi

CKPT_DIR="$REPO/checkpoints/track_a/videomaev2+ft"
RESUME_PT="${CKPT_DIR}/${RUN}.last.pt"
if [[ ! -f "$RESUME_PT" ]]; then
  RESUME_PT="${CKPT_DIR}/${RUN}.pt"
fi
if [[ ! -f "$RESUME_PT" ]]; then
  echo "ABORT: no checkpoint for $RUN" >&2
  exit 1
fi

# Stop stale trainer (VM reboot may leave dead PIDs; kill matching train cmd).
pkill -f "training.wandb.name=${RUN}" 2>/dev/null || true
pkill -f "training.checkpoint_path=.*${RUN}.pt" 2>/dev/null || true
sleep 2

if pgrep -f "training.wandb.name=${RUN}" >/dev/null 2>&1; then
  echo "ABORT: trainer still running for $RUN" >&2
  exit 1
fi

{
  echo ""
  echo "# === RESUME $(date -Is) ==="
  echo "# resume_from: ${RESUME_PT}"
  echo "# wandb_run_id: ${WID}"
  echo "# wandb_resume: allow"
} >>"$LOG"

if [[ "$KIND" == meanpool ]]; then
  SEED=44
  SSL="$REPO/checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep50.pt"
  EXP=track_a_videomae_official_ssv2_ft
  EXTRA=(
    "seed=${SEED}"
    "model.init_from=${SSL}"
    model.tube_t=1 dataset.num_frames=4
    dataset.include_val_in_train=false dataset.official_val_holdout_ratio=0
    "training.checkpoint_path=${CKPT_DIR}/${RUN}.pt"
    "training.resume_from=${RESUME_PT}"
    "training.wandb.project=smth2smth-frame-ablation"
    ++training.wandb.group=ft-mae-scaling
    "training.wandb.name=${RUN}"
    ++training.wandb.config.head_type=meanpool
    ++training.wandb.config.pretrain_epochs=50
  )
else
  SEED=$(echo "$RUN" | sed 's/.*-s//')
  Q=$(echo "$RUN" | sed 's/perceiverQ//' | cut -d- -f1)
  SSL="$REPO/checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep500.pt"
  EXP=track_a_diverse_arch2_perceiver_stab
  EXTRA=(
    "seed=${SEED}"
    "model.init_from=${SSL}"
    "model.head_queries=${Q}"
    model.tube_t=1 dataset.num_frames=4
    dataset.include_val_in_train=false dataset.official_val_holdout_ratio=0
    "training.checkpoint_path=${CKPT_DIR}/${RUN}.pt"
    "training.resume_from=${RESUME_PT}"
    "training.wandb.project=smth2smth-frame-ablation"
    ++training.wandb.group=ft-mae-scaling
    "training.wandb.name=${RUN}"
    ++training.wandb.config.head_type="perceiverQ${Q}"
    ++training.wandb.config.pretrain_epochs=500
    ++training.wandb.config.num_queries="${Q}"
  )
fi

test -f "$SSL" || { echo "ABORT: missing SSL $SSL" >&2; exit 1; }

nohup env PYTHONPATH=src PYTHONUNBUFFERED=1 WANDB_MODE=online \
  WANDB_RESUME=allow "WANDB_RUN_ID=${WID}" \
  .venv/bin/python -u -m smth2smth.pipelines.train \
  track=a "experiment=${EXP}" \
  "${EXTRA[@]}" \
  >>"$LOG" 2>&1 &
echo "trainer_pid=$! log=$LOG resume=$RESUME_PT"
REMOTE
}

health_one() {
  local H="$1" RUN="$2"
  local LOG
  LOG=$(ssh -o BatchMode=yes "$H" "ls -t $REPO/logs/track_a/${RUN}_*.log 2>/dev/null | head -1")
  echo "=== HEALTH $H $RUN ==="
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$LOG" <<'HEALTH'
set -euo pipefail
REPO="$1" LOG="$2"
cd "$REPO"
test -f "$LOG"
tail -15 "$LOG"
grep -q Traceback "$LOG" && exit 1
grep -q 'Resuming from checkpoint:' "$LOG" && echo "resume_line=ok" || echo "resume_line=missing"
grep -q '\[wandb\] run started:' "$LOG" && echo "wandb=ok" || echo "wandb=missing"
grep -qE '\[.*\] step [0-9]+/' "$LOG" || exit 1
echo OK
HEALTH
}

case "$MODE" in
  resume)
    for e in "${ENTRIES[@]}"; do
      IFS=: read -r H RUN WID KIND <<< "$e"
      resume_one "$H" "$RUN" "$WID" "$KIND" || echo "FAIL $H $RUN"
    done
    echo "Waiting 90s..."
    sleep 90
    ;;
  health)
    ok=0 fail=0
    for e in "${ENTRIES[@]}"; do
      IFS=: read -r H RUN WID KIND <<< "$e"
      if health_one "$H" "$RUN"; then ok=$((ok+1)); else fail=$((fail+1)); fi
    done
    echo "health: ok=$ok fail=$fail"
    ;;
  *)
    echo "Usage: $0 [resume|health]"; exit 1 ;;
esac
