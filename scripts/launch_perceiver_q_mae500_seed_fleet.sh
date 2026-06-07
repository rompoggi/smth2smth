#!/usr/bin/env bash
# Perceiver Q seed sweep @ MAE500 SSL (train-only honest val): Q in {2,4,8,32,64}, seeds 42–44.
# Coordinator: gymnote. Usage: bash scripts/launch_perceiver_q_mae500_seed_fleet.sh [copy|prep|launch|health|all]
set -euo pipefail

REPO="${REPO:-/Data/romain.poggi/smth2smth}"
cd "$REPO"
SRC_HOST="${SRC_HOST:-gymnote}"
TAG="${TAG:-$(date +%Y%m%d)}"
MODE="${1:-all}"
SSL_EP=500
SSL_REL="checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep${SSL_EP}.pt"
EXPERIMENT=track_a_diverse_arch2_perceiver_stab

# 11 runs on 11 worker hosts (gymnote = coordinator only; truite spare).
HOSTS=(gardon lieu lotte mulet murene piranha raie requin roussette sole thon)
RUNS=(
  perceiverQ2-mae500-s43 perceiverQ2-mae500-s44
  perceiverQ4-mae500-s43 perceiverQ4-mae500-s44
  perceiverQ8-mae500-s43 perceiverQ8-mae500-s44
  perceiverQ32-mae500-s43 perceiverQ32-mae500-s44
  perceiverQ64-mae500-s42 perceiverQ64-mae500-s43 perceiverQ64-mae500-s44
)

parse_q() {
  local run="$1"
  echo "${run#perceiverQ}" | cut -d- -f1
}

parse_seed() {
  local run="$1"
  echo "${run##*-s}"
}

ssl_path() {
  echo "${REPO}/${SSL_REL}"
}

copy_ssl_one() {
  local H="$1"
  local dest="${REPO}/${SSL_REL}"
  local here
  here="$(hostname -s 2>/dev/null || hostname)"
  test -f "$dest" || { echo "MISSING on coordinator: $dest" >&2; return 1; }
  if [[ "$H" == "$SRC_HOST" || "$H" == "$here" ]]; then
    echo "OK $H ep${SSL_EP} (local)"
    return 0
  fi
  ssh -o BatchMode=yes "$H" "mkdir -p ${REPO}/checkpoints/track_a/ssl/pretrain"
  rsync -az "$dest" "${H}:${dest}"
  ssh -o BatchMode=yes "$H" "test -f '${dest}' && ls -lh '${dest}'"
}

copy_all() {
  for H in "${HOSTS[@]}"; do
    echo "=== COPY ep${SSL_EP} -> $H ==="
    copy_ssl_one "$H" || echo "FAIL copy $H"
  done
}

sync_code() {
  local H="$1"
  local here
  here="$(hostname -s 2>/dev/null || hostname)"
  if [[ "$H" == "$here" ]]; then
    return 0
  fi
  rsync -az "${REPO}/src/" "${H}:${REPO}/src/"
  rsync -az "${REPO}/configs/" "${H}:${REPO}/configs/"
  rsync -az "${REPO}/scripts/launch_perceiver_q_mae500_seed_fleet.sh" "${H}:${REPO}/scripts/"
}

prep_host() {
  local H="$1"
  sync_code "$H"
  ssh -o BatchMode=yes "$H" bash -s <<PREP
set -euo pipefail
cd "${REPO}"
git pull --ff-only 2>/dev/null || git pull 2>/dev/null || true
test -x .venv/bin/python || uv sync
test -d data/train || { echo "no data/train"; exit 1; }
PREP
}

launch_one() {
  local H="$1" RUN="$2"
  local Q SEED HEAD SSL LOG
  Q="$(parse_q "$RUN")"
  SEED="$(parse_seed "$RUN")"
  HEAD="perceiverQ${Q}"
  SSL="$(ssl_path)"
  LOG="logs/track_a/${RUN}_${TAG}.log"
  echo "=== LAUNCH $H -> $RUN (Q=${Q}, seed=${SEED}) ==="
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$RUN" "$SSL" "$Q" "$SEED" "$HEAD" "$EXPERIMENT" "$LOG" <<'LAUNCH'
set -euo pipefail
REPO="$1" RUN="$2" SSL="$3" Q="$4" SEED="$5" HEAD_TYPE="$6" EXP="$7" LOG="$8"
cd "$REPO"
mkdir -p logs/track_a checkpoints/track_a/videomaev2+ft checkpoints/track_a/ssl/pretrain
if pgrep -af "training.wandb.name=${RUN}" >/dev/null 2>&1; then
  echo "SKIP: ${RUN} already running on $(hostname)"
  exit 0
fi
if pgrep -af "training.checkpoint_path=.*${RUN}.pt" >/dev/null 2>&1; then
  echo "SKIP: ckpt ${RUN} already training on $(hostname)"
  exit 0
fi
test -f "$SSL" || { echo "error: missing SSL $SSL" >&2; exit 1; }
: >"$LOG"
{
  echo "# run: ${RUN}"
  echo "# started: $(date -Is)"
  echo "# track: a"
  echo "# experiment_doc: experiments/diverse_classifier_heads_post_mae.md"
  echo "# hydra: experiment=${EXP} seed=${SEED} model.head_queries=${Q} T=4 train-only"
  echo "# host: $(hostname)"
  echo "# ssl: ${SSL}"
  echo "# wandb_name: ${RUN}"
} >>"$LOG"
nohup env PYTHONPATH=src PYTHONUNBUFFERED=1 WANDB_MODE=online \
  .venv/bin/python -u -m smth2smth.pipelines.train \
  track=a "experiment=${EXP}" "seed=${SEED}" \
  "model.init_from=${SSL}" "model.head_queries=${Q}" model.tube_t=1 dataset.num_frames=4 \
  dataset.include_val_in_train=false dataset.official_val_holdout_ratio=0 \
  "training.checkpoint_path=checkpoints/track_a/videomaev2+ft/${RUN}.pt" \
  "training.wandb.project=smth2smth-frame-ablation" \
  ++training.wandb.group=ft-mae-scaling \
  "training.wandb.name=${RUN}" \
  ++training.wandb.config.head_type="${HEAD_TYPE}" \
  ++training.wandb.config.pretrain_epochs=500 \
  ++training.wandb.config.num_queries="${Q}" \
  >>"$LOG" 2>&1 &
echo "trainer_pid=$! log=$LOG"
LAUNCH
}

health_one() {
  local H="$1" RUN="$2"
  local LOG="logs/track_a/${RUN}_${TAG}.log"
  echo "=== HEALTH $H $RUN ==="
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$LOG" <<'HEALTH'
set -euo pipefail
REPO="$1" LOG="$2"
cd "$REPO"
test -f "$LOG" || { echo "FAIL: no log"; exit 1; }
tail -12 "$LOG"
grep -q Traceback "$LOG" && { echo "FAIL: traceback"; exit 1; }
grep -q '\[init_from\] loaded' "$LOG" || { echo "FAIL: no init_from"; exit 1; }
grep -q '\[wandb\] run started:' "$LOG" || { echo "FAIL: no wandb"; exit 1; }
grep -qE '\[.*\] step [0-9]+/' "$LOG" || { echo "FAIL: no step lines"; exit 1; }
if LC_ALL=C grep -P '[^\x00-\x7F]' "$LOG" | grep -qE 'step |Epoch'; then
  echo "FAIL: non-ASCII in training lines"
  exit 1
fi
echo "OK"
HEALTH
}

case "$MODE" in
  copy) copy_all ;;
  prep)
    for H in "${HOSTS[@]}"; do prep_host "$H" || echo "FAIL prep $H"; done
    ;;
  launch)
    for i in "${!HOSTS[@]}"; do
      launch_one "${HOSTS[$i]}" "${RUNS[$i]}"
    done
    ;;
  health)
    ok=0 fail=0
    for i in "${!HOSTS[@]}"; do
      if health_one "${HOSTS[$i]}" "${RUNS[$i]}"; then ok=$((ok+1)); else fail=$((fail+1)); fi
    done
    echo "health: ok=$ok fail=$fail"
    ;;
  all)
    copy_all
    for H in "${HOSTS[@]}"; do prep_host "$H" || exit 1; done
    for i in "${!HOSTS[@]}"; do
      launch_one "${HOSTS[$i]}" "${RUNS[$i]}"
    done
    echo "Waiting 90s for boot..."
    sleep 90
    health_one "${HOSTS[0]}" "${RUNS[0]}" || true
    ;;
  *)
    echo "Usage: $0 [copy|prep|launch|health|all]"; exit 1
    ;;
esac
echo "Done ($MODE)."
