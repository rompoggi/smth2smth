#!/usr/bin/env bash
# Perceiver Q16 @ odd SSL epochs {50,150,250,350,450} × seeds {42,43,44} (15 runs).
# Coordinator: gymnote. Usage: bash scripts/launch_perceiverQ16_mae_odd_seed_fleet.sh [copy|prep|launch|health|all]
set -euo pipefail

REPO="${REPO:-/Data/romain.poggi/smth2smth}"
cd "$REPO"
SRC_HOST="${SRC_HOST:-gymnote}"
TAG="${TAG:-$(date +%Y%m%d)}"
MODE="${1:-all}"
EXPERIMENT=track_a_diverse_arch2_perceiver

# One run per host (16 listed; gymnote = coordinator, 15 workers below).
HOSTS=(ablette anchois anguille barbeau barbue baudroie carrelet labre rouget saumon silure truite gymnote mulet murene)
RUNS=(
  perceiverQ16-mae050-s42 perceiverQ16-mae050-s43 perceiverQ16-mae050-s44
  perceiverQ16-mae150-s42 perceiverQ16-mae150-s43 perceiverQ16-mae150-s44
  perceiverQ16-mae250-s42 perceiverQ16-mae250-s43 perceiverQ16-mae250-s44
  perceiverQ16-mae350-s42 perceiverQ16-mae350-s43 perceiverQ16-mae350-s44
  perceiverQ16-mae450-s42 perceiverQ16-mae450-s43 perceiverQ16-mae450-s44
)
SSL_EPOCHS=(50 50 50 150 150 150 250 250 250 350 350 350 450 450 450)
SEEDS=(42 43 44 42 43 44 42 43 44 42 43 44 42 43 44)

parse_seed() { echo "${1##*-s}"; }

ssl_path() {
  local ep="$1"
  echo "${REPO}/checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep${ep}.pt"
}

host_busy() {
  local H="$1"
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$H" \
    "pgrep -af 'smth2smth.pipelines.train' | grep -v pgrep | grep -q ." 2>/dev/null
}

copy_ssl_one() {
  local H="$1" ep="$2"
  local rel="checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep${ep}.pt"
  local dest="${REPO}/${rel}"
  local here
  here="$(hostname -s 2>/dev/null || hostname)"
  test -f "$dest" || { echo "MISSING on coordinator: $dest" >&2; return 1; }
  if [[ "$H" == "$SRC_HOST" || "$H" == "$here" ]]; then
    echo "OK $H ep${ep} (local)"
    return 0
  fi
  ssh -o BatchMode=yes "$H" "mkdir -p ${REPO}/checkpoints/track_a/ssl/pretrain"
  rsync -az "$dest" "${H}:${dest}"
  ssh -o BatchMode=yes "$H" "test -f '${dest}' && ls -lh '${dest}'"
}

copy_all() {
  local i ep
  for i in "${!HOSTS[@]}"; do
    ep="${SSL_EPOCHS[$i]}"
    echo "=== COPY ep${ep} -> ${HOSTS[$i]} ==="
    copy_ssl_one "${HOSTS[$i]}" "$ep" || echo "FAIL copy ${HOSTS[$i]} ep${ep}"
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
  local H="$1" RUN="$2" ep="$3" seed="$4"
  local SSL LOG
  SSL="$(ssl_path "$ep")"
  LOG="logs/track_a/${RUN}_${TAG}.log"
  echo "=== LAUNCH $H -> $RUN (ep${ep}, seed=${seed}) ==="
  if host_busy "$H"; then
    echo "SKIP: $H has an active trainer (free GPU first)"
    return 1
  fi
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$RUN" "$SSL" "$ep" "$seed" "$EXPERIMENT" "$LOG" <<'LAUNCH'
set -euo pipefail
REPO="$1" RUN="$2" SSL="$3" PRETRAIN_EPOCHS="$4" SEED="$5" EXP="$6" LOG="$7"
cd "$REPO"
mkdir -p logs/track_a checkpoints/track_a/videomaev2+ft checkpoints/track_a/ssl/pretrain
if pgrep -af "training.wandb.name=${RUN}" >/dev/null 2>&1; then
  echo "SKIP: ${RUN} already running on $(hostname)"
  exit 0
fi
test -f "$SSL" || { echo "error: missing SSL $SSL" >&2; exit 1; }
: >"$LOG"
{
  echo "# run: ${RUN}"
  echo "# started: $(date -Is)"
  echo "# track: a"
  echo "# experiment_doc: experiments/experimental_cleanup.md"
  echo "# hydra: experiment=${EXP} seed=${SEED} model.head_queries=16 T=4 train-only"
  echo "# host: $(hostname)"
  echo "# ssl: ${SSL}"
  echo "# wandb_name: ${RUN}"
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
  ++training.wandb.config.pretrain_epochs="${PRETRAIN_EPOCHS}" \
  ++training.wandb.config.num_queries=16 \
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
echo "OK"
HEALTH
}

case "$MODE" in
  copy) copy_all ;;
  prep)
    for H in "${HOSTS[@]}"; do prep_host "$H" || echo "FAIL prep $H"; done
    ;;
  launch)
    ok=0 fail=0
    for i in "${!HOSTS[@]}"; do
      if launch_one "${HOSTS[$i]}" "${RUNS[$i]}" "${SSL_EPOCHS[$i]}" "${SEEDS[$i]}"; then
        ok=$((ok + 1))
      else
        fail=$((fail + 1))
      fi
    done
    echo "launch: ok=$ok skip_or_fail=$fail"
    ;;
  health)
    ok=0 fail=0 skip=0
    for i in "${!HOSTS[@]}"; do
      if host_busy "${HOSTS[$i]}" && [[ ! -f "${REPO}/logs/track_a/${RUNS[$i]}_${TAG}.log" ]]; then
        echo "SKIP health ${HOSTS[$i]} (no log, host was busy)"
        skip=$((skip + 1))
      elif health_one "${HOSTS[$i]}" "${RUNS[$i]}"; then ok=$((ok + 1)); else fail=$((fail + 1)); fi
    done
    echo "health: ok=$ok fail=$fail skip=$skip"
    ;;
  all)
    copy_all
    for H in "${HOSTS[@]}"; do prep_host "$H" || echo "FAIL prep $H"; done
    ok=0 fail=0
    for i in "${!HOSTS[@]}"; do
      if launch_one "${HOSTS[$i]}" "${RUNS[$i]}" "${SSL_EPOCHS[$i]}" "${SEEDS[$i]}"; then
        ok=$((ok + 1))
      else
        fail=$((fail + 1))
      fi
    done
    echo "launch: ok=$ok skip_or_fail=$fail"
    echo "Waiting 90s..."
    sleep 90
    health_one "${HOSTS[0]}" "${RUNS[0]}" || true
    ;;
  *)
    echo "Usage: $0 [copy|prep|launch|health|all]"; exit 1
    ;;
esac
echo "Done ($MODE)."
