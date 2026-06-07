#!/usr/bin/env bash
# Resume stopped Track A fleet on free hosts + start perceiverQ16-mae450-s43/s44 (replace DivSpaceTimeK1 on murene).
# Usage: bash scripts/resume_free_fleet_20260602.sh [resume-mae450|resume-all|health]
set -euo pipefail

REPO="${REPO:-/Data/romain.poggi/smth2smth}"
TAG_Q16="${TAG_Q16:-20260601}"
TAG_MAE500="${TAG_MAE500:-20260531}"
MODE="${1:-resume-all}"

# host:run:wandb_id:experiment_kind  (kind = q16_odd | mae500_stab)
Q16_ENTRIES=(
  ablette:perceiverQ16-mae050-s42:fvdablod:q16_odd
  anguille:perceiverQ16-mae050-s44:rubhtr2m:q16_odd
  barbeau:perceiverQ16-mae150-s42:op2nj6cb:q16_odd
  barbue:perceiverQ16-mae150-s43:qrm8svyz:q16_odd
  baudroie:perceiverQ16-mae150-s44:jtgu5c4e:q16_odd
  carrelet:perceiverQ16-mae250-s42:dozqo4kv:q16_odd
  labre:perceiverQ16-mae250-s43:mvvlz1an:q16_odd
  rouget:perceiverQ16-mae250-s44:tm2vwp8i:q16_odd
  saumon:perceiverQ16-mae350-s42:scxr4ydc:q16_odd
  silure:perceiverQ16-mae350-s43:jryp4bz7:q16_odd
  truite:perceiverQ16-mae350-s44:ggd2sux0:q16_odd
  gymnote:perceiverQ16-mae450-s42:e319tzm1:q16_odd
)

MAE500_ENTRIES=(
  gardon:perceiverQ2-mae500-s43:7ew6eok8:mae500_stab
  lotte:perceiverQ4-mae500-s43:ldvozwwn:mae500_stab
  piranha:perceiverQ8-mae500-s44:2wusde6o:mae500_stab
  raie:perceiverQ32-mae500-s43:20ykyagi:mae500_stab
  requin:perceiverQ32-mae500-s44:532pq1s0:mae500_stab
  roussette:perceiverQ64-mae500-s42:e4h78uqj:mae500_stab
  sole:perceiverQ64-mae500-s43:0xxh9rr3:mae500_stab
)

ckpt_global_step() {
  local H="$1" RUN="$2"
  ssh -o BatchMode=yes "$H" "$REPO/.venv/bin/python" - "$RUN" "$REPO" <<'PY'
import torch, sys
run, repo = sys.argv[1], sys.argv[2]
for name in (f"{run}.last.pt", f"{run}.pt"):
    p = f"{repo}/checkpoints/track_a/videomaev2+ft/{name}"
    try:
        c = torch.load(p, map_location="cpu", weights_only=False)
    except FileNotFoundError:
        continue
    gs = (c.get("extra") or {}).get("global_step")
    if gs is not None:
        print(int(gs))
        raise SystemExit(0)
raise SystemExit(1)
PY
}

stop_divspace_k1() {
  echo "=== STOP DivSpaceTimeK1-mae500-s42 on murene ==="
  ssh -o BatchMode=yes murene bash -s <<'STOP'
set -euo pipefail
REPO=/Data/romain.poggi/smth2smth
RUN=DivSpaceTimeK1-mae500-s42
pkill -f "training.wandb.name=${RUN}" 2>/dev/null || true
sleep 3
if pgrep -f "training.wandb.name=${RUN}" >/dev/null 2>&1; then
  echo "FAIL: ${RUN} still running on $(hostname -s)" >&2
  exit 1
fi
LOG=$(ls -t "$REPO/logs/track_a/divspace_s42/${RUN}"_*.log 2>/dev/null | head -1)
if [[ -n "${LOG:-}" ]]; then
  {
    echo ""
    echo "# === STOPPED for perceiverQ16-mae450-s44 $(date -Is) ==="
  } >>"$LOG"
fi
echo "stopped ${RUN}"
STOP
}

launch_mae450_fresh() {
  local H="$1" RUN="$2" SEED="$3"
  local SSL="${REPO}/checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep450.pt"
  local LOG="logs/track_a/${RUN}_${TAG_Q16}.log"
  echo "=== FRESH LAUNCH $H $RUN seed=$SEED ==="
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$RUN" "$SSL" "$SEED" "$LOG" <<'LAUNCH'
set -euo pipefail
REPO="$1" RUN="$2" SSL="$3" SEED="$4" LOG="$5"
cd "$REPO"
EXP=track_a_diverse_arch2_perceiver
mkdir -p logs/track_a checkpoints/track_a/videomaev2+ft checkpoints/track_a/ssl/pretrain
pkill -f "training.wandb.name=${RUN}" 2>/dev/null || true
sleep 2
if pgrep -af "training.wandb.name=${RUN}" | grep -q python; then
  echo "ABORT: already running" >&2
  exit 1
fi
test -f "$SSL" || { echo "ABORT: missing $SSL" >&2; exit 1; }
if [[ ! -f "$LOG" ]]; then
  {
    echo "# run: ${RUN}"
    echo "# started: $(date -Is)"
    echo "# track: a"
    echo "# experiment_doc: experiments/experimental_cleanup.md"
    echo "# hydra: experiment=${EXP} seed=${SEED} model.head_queries=16 T=4 train-only SSL ep450"
  } >"$LOG"
else
  {
    echo ""
    echo "# === FRESH START $(date -Is) (replaced DivSpaceTimeK1 on murene for s44) ==="
  } >>"$LOG"
fi
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
  ++training.wandb.config.pretrain_epochs=450 \
  ++training.wandb.config.num_queries=16 \
  >>"$LOG" 2>&1 &
echo "trainer_pid=$! log=$LOG"
LAUNCH
}

resume_one() {
  local H="$1" RUN="$2" WID="$3" KIND="$4"
  local TAG="$TAG_Q16"
  local EXP=track_a_diverse_arch2_perceiver
  local SSL_EP=50
  if [[ "$KIND" == mae500_stab ]]; then
    TAG="$TAG_MAE500"
    EXP=track_a_diverse_arch2_perceiver_stab
    SSL_EP=500
  else
    case "$RUN" in
      *-mae050-*) SSL_EP=50 ;;
      *-mae150-*) SSL_EP=150 ;;
      *-mae250-*) SSL_EP=250 ;;
      *-mae350-*) SSL_EP=350 ;;
      *-mae450-*) SSL_EP=450 ;;
      *) SSL_EP=500 ;;
    esac
  fi

  local GSTEP
  GSTEP=$(ckpt_global_step "$H" "$RUN") || {
    echo "FAIL: no global_step in ckpt for $H $RUN"
    return 1
  }

  echo "=== RESUME $H $RUN wandb=$WID global_step=$GSTEP ==="
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$RUN" "$WID" "$KIND" "$TAG" "$EXP" "$SSL_EP" "$GSTEP" <<'REMOTE'
set -euo pipefail
REPO="$1" RUN="$2" WID="$3" KIND="$4" TAG="$5" EXP="$6" SSL_EP="$7" GSTEP="$8"
cd "$REPO"

LOG=$(ls -t "$REPO/logs/track_a/${RUN}_${TAG}.log" 2>/dev/null | head -1)
if [[ -z "$LOG" ]]; then
  LOG="logs/track_a/${RUN}_${TAG}.log"
fi

CKPT_DIR="$REPO/checkpoints/track_a/videomaev2+ft"
RESUME_PT="${CKPT_DIR}/${RUN}.last.pt"
[[ -f "$RESUME_PT" ]] || RESUME_PT="${CKPT_DIR}/${RUN}.pt"
[[ -f "$RESUME_PT" ]] || { echo "ABORT: no checkpoint" >&2; exit 1; }

pkill -f "training.wandb.name=${RUN}" 2>/dev/null || true
sleep 2
if pgrep -af "training.wandb.name=${RUN}" | grep -q python; then
  echo "ABORT: still running" >&2
  exit 1
fi

SSL="$REPO/checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep${SSL_EP}.pt"
test -f "$SSL" || { echo "ABORT: missing $SSL" >&2; exit 1; }

{
  echo ""
  echo "# === RESUME $(date -Is) ==="
  echo "# resume_from: ${RESUME_PT}"
  echo "# resume_global_step: ${GSTEP}"
  echo "# wandb_run_id: ${WID}"
} >>"$LOG"

SEED="${RUN##*-s}"
Q=""
if [[ "$KIND" == mae500_stab ]]; then
  Q=$(echo "$RUN" | sed 's/perceiverQ//' | cut -d- -f1)
fi

ARGS=(
  track=a "experiment=${EXP}" "seed=${SEED}"
  "model.init_from=${SSL}"
  model.tube_t=1 dataset.num_frames=4
  dataset.include_val_in_train=false dataset.official_val_holdout_ratio=0
  "training.checkpoint_path=${CKPT_DIR}/${RUN}.pt"
  "training.resume_from=${RESUME_PT}"
  "training.resume_global_step=${GSTEP}"
  "training.wandb.project=smth2smth-frame-ablation"
  ++training.wandb.group=ft-mae-scaling
  "training.wandb.name=${RUN}"
  ++training.wandb.config.pretrain_epochs="${SSL_EP}"
)
if [[ "$KIND" == mae500_stab ]]; then
  ARGS+=("model.head_queries=${Q}" ++training.wandb.config.head_type="perceiverQ${Q}" ++training.wandb.config.num_queries="${Q}")
else
  ARGS+=(model.head_queries=16 ++training.wandb.config.head_type=perceiverQ16 ++training.wandb.config.num_queries=16)
fi

nohup env PYTHONPATH=src PYTHONUNBUFFERED=1 WANDB_MODE=online \
  WANDB_RESUME=allow "WANDB_RUN_ID=${WID}" \
  .venv/bin/python -u -m smth2smth.pipelines.train \
  "${ARGS[@]}" \
  >>"$LOG" 2>&1 &
echo "trainer_pid=$! log=$LOG resume_step=$GSTEP"
REMOTE
}

health_one() {
  local H="$1" RUN="$2" TAG="${3:-$TAG_Q16}"
  local LOG="logs/track_a/${RUN}_${TAG}.log"
  echo "=== HEALTH $H $RUN ==="
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$LOG" <<'HEALTH'
set -euo pipefail
REPO="$1" LOG="$2"
cd "$REPO"
pgrep -af "smth2smth.pipelines.train" | grep python | head -1 || { echo "FAIL: no trainer"; exit 1; }
test -f "$LOG"
tail -8 "$LOG"
grep -q Traceback "$LOG" && { echo "FAIL: traceback"; exit 1; }
grep -qE '\[.*\] step [0-9]+/' "$LOG" || grep -q 'Epoch [0-9]+/' "$LOG" || { echo "FAIL: no progress"; exit 1; }
echo OK
HEALTH
}

case "$MODE" in
  resume-mae450)
    stop_divspace_k1
    launch_mae450_fresh mulet perceiverQ16-mae450-s43 43
    launch_mae450_fresh murene perceiverQ16-mae450-s44 44
    ;;
  resume-all)
    stop_divspace_k1
    launch_mae450_fresh mulet perceiverQ16-mae450-s43 43
    launch_mae450_fresh murene perceiverQ16-mae450-s44 44
    for e in "${Q16_ENTRIES[@]}"; do
      IFS=: read -r H RUN WID KIND <<<"$e"
      resume_one "$H" "$RUN" "$WID" "$KIND" || echo "FAIL $H $RUN"
    done
    for e in "${MAE500_ENTRIES[@]}"; do
      IFS=: read -r H RUN WID KIND <<<"$e"
      resume_one "$H" "$RUN" "$WID" "$KIND" || echo "FAIL $H $RUN"
    done
    echo "Waiting 90s for health..."
    sleep 90
    ;;
  resume-fleet)
    for e in "${Q16_ENTRIES[@]}"; do
      IFS=: read -r H RUN WID KIND <<<"$e"
      resume_one "$H" "$RUN" "$WID" "$KIND" || echo "FAIL $H $RUN"
    done
    for e in "${MAE500_ENTRIES[@]}"; do
      IFS=: read -r H RUN WID KIND <<<"$e"
      resume_one "$H" "$RUN" "$WID" "$KIND" || echo "FAIL $H $RUN"
    done
    echo "Waiting 90s for health..."
    sleep 90
    ;;
  health)
    health_one mulet perceiverQ16-mae450-s43 || true
    health_one murene perceiverQ16-mae450-s44 || true
    for e in "${Q16_ENTRIES[@]}" "${MAE500_ENTRIES[@]}"; do
      IFS=: read -r H RUN WID KIND <<<"$e"
      TAG="$TAG_Q16"
      [[ "$KIND" == mae500_stab ]] && TAG="$TAG_MAE500"
      health_one "$H" "$RUN" "$TAG" || true
    done
    ;;
  *)
    echo "Usage: $0 [resume-mae450|resume-fleet|resume-all|health]"
    exit 1
    ;;
esac
