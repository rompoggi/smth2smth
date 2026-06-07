#!/usr/bin/env bash
# Seed-43 MAE scaling fine-tune fleet (15 hosts): mean-pool curve + Perceiver Q16 ablation.
# Coordinator: gymnote. Run: bash scripts/launch_ft_mae_scaling_s43_fleet.sh [copy|canary|launch|health|all]
set -euo pipefail

REPO="${REPO:-/Data/romain.poggi/smth2smth}"
cd "$REPO"
SRC_HOST="${SRC_HOST:-gymnote}"
TAG="${TAG:-$(date +%Y%m%d)}"
MODE="${1:-all}"

HOSTS=(gardon gymnote labre lieu lotte mulet murene piranha raie requin rouget roussette sole thon truite)
# Local run / checkpoint basename (under videomaev2+ft/)
RUNS=(
  meanpool-mae050-s43 meanpool-mae100-s43 meanpool-mae150-s43 meanpool-mae200-s43 meanpool-mae250-s43
  meanpool-mae300-s43 meanpool-mae350-s43 meanpool-mae400-s43 meanpool-mae450-s43 meanpool-mae500-s43
  perceiverQ16-mae100-s43 perceiverQ16-mae200-s43 perceiverQ16-mae300-s43 perceiverQ16-mae400-s43 perceiverQ16-mae500-s43
)
WANDB_NAMES=("${RUNS[@]}")
SSL_EPOCHS=(50 100 150 200 250 300 350 400 450 500 100 200 300 400 500)
HEAD_TYPES=(meanpool meanpool meanpool meanpool meanpool meanpool meanpool meanpool meanpool meanpool \
  perceiverQ16 perceiverQ16 perceiverQ16 perceiverQ16 perceiverQ16)
EXPERIMENTS=(
  track_a_videomae_official_ssv2_ft track_a_videomae_official_ssv2_ft track_a_videomae_official_ssv2_ft
  track_a_videomae_official_ssv2_ft track_a_videomae_official_ssv2_ft track_a_videomae_official_ssv2_ft
  track_a_videomae_official_ssv2_ft track_a_videomae_official_ssv2_ft track_a_videomae_official_ssv2_ft
  track_a_videomae_official_ssv2_ft
  track_a_diverse_arch2_perceiver track_a_diverse_arch2_perceiver track_a_diverse_arch2_perceiver
  track_a_diverse_arch2_perceiver track_a_diverse_arch2_perceiver
)

ssl_path() {
  local ep="$1"
  echo "${REPO}/checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep${ep}.pt"
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
  if [[ "$H" == "$(hostname -s)" || "$H" == "$(hostname)" ]]; then
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
  local H="$1" RUN="$2" WNAME="$3" ep="$4" head="$5" EXP="$6" canary="${7:-0}"
  local SSL
  SSL="$(ssl_path "$ep")"
  local LOG="logs/track_a/${RUN}_${TAG}.log"
  if [[ "$canary" == "1" ]]; then
    LOG="logs/track_a/canary-${RUN}_${TAG}.log"
    RUN="canary-${RUN}"
  fi
  echo "=== LAUNCH $H -> $RUN (ep${ep}, $head) ==="
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$RUN" "$WNAME" "$SSL" "$ep" "$head" "$EXP" "$LOG" "$canary" <<'LAUNCH'
set -euo pipefail
REPO="$1"
RUN="$2"
WNAME="$3"
SSL="$4"
PRETRAIN_EPOCHS="$5"
HEAD_TYPE="$6"
EXP="$7"
LOG="$8"
CANARY="$9"
cd "$REPO"
mkdir -p logs/track_a checkpoints/track_a/videomaev2+ft checkpoints/track_a/ssl/pretrain
if pgrep -af "experiment=${EXP}.*seed=43" >/dev/null 2>&1; then
  echo "SKIP: seed=43 ${EXP} already running on $(hostname)"
  exit 0
fi
if [[ ! -f "$SSL" ]]; then
  echo "error: missing SSL $SSL" >&2
  exit 1
fi
if [[ -f "$LOG" ]]; then
  : >"$LOG"
fi
{
  echo "# run: ${RUN}"
  echo "# started: $(date -Is)"
  echo "# track: a"
  echo "# experiment_doc: experiments/experimental_cleanup.md"
  echo "# hydra: experiment=${EXP} seed=43 T=4"
  echo "# host: $(hostname)"
  echo "# ssl: ${SSL}"
  echo "# wandb_name: ${WNAME}"
} >>"$LOG"
EXTRA=()
if [[ "$CANARY" == "1" ]]; then
  EXTRA+=(dataset.max_samples=256 training.epochs=1)
fi
nohup env PYTHONPATH=src PYTHONUNBUFFERED=1 WANDB_MODE=online \
  .venv/bin/python -u -m smth2smth.pipelines.train \
  track=a "experiment=${EXP}" seed=43 \
  "model.init_from=${SSL}" model.tube_t=1 dataset.num_frames=4 \
  dataset.include_val_in_train=false dataset.official_val_holdout_ratio=0 \
  "training.checkpoint_path=checkpoints/track_a/videomaev2+ft/${RUN}.pt" \
  "training.wandb.project=smth2smth-frame-ablation" \
  ++training.wandb.group=ft-mae-scaling \
  "training.wandb.name=${WNAME}" \
  ++training.wandb.config.head_type="${HEAD_TYPE}" \
  ++training.wandb.config.pretrain_epochs="${PRETRAIN_EPOCHS}" \
  "${EXTRA[@]}" \
  >>"$LOG" 2>&1 &
echo "trainer_pid=$! log=$LOG"
LAUNCH
}

health_one() {
  local H="$1" RUN="$2"
  local LOG="logs/track_a/canary-${RUN}_${TAG}.log"
  echo "=== HEALTH $H $LOG ==="
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$LOG" <<'HEALTH'
set -euo pipefail
REPO="$1"
LOG="$2"
cd "$REPO"
if [[ ! -f "$LOG" ]]; then echo "FAIL: no log"; exit 1; fi
tail -20 "$LOG"
if grep -q Traceback "$LOG"; then echo "FAIL: traceback"; exit 1; fi
if ! grep -q '\[init_from\] loaded' "$LOG"; then echo "FAIL: no init_from"; exit 1; fi
if ! grep -q '\[wandb\] run started:' "$LOG"; then echo "FAIL: no wandb"; exit 1; fi
if ! grep -qE '\[.*\] step [0-9]+/' "$LOG"; then echo "FAIL: no step lines"; exit 1; fi
if LC_ALL=C grep -P '[^\x00-\x7F]' "$LOG" | grep -qE 'step |Epoch'; then echo "FAIL: non-ASCII in steps"; exit 1; fi
echo "OK"
HEALTH
}

case "$MODE" in
  copy) copy_all ;;
  prep)
    for H in "${HOSTS[@]}"; do prep_host "$H" || echo "FAIL prep $H"; done
    ;;
  canary)
    for i in "${!HOSTS[@]}"; do
      launch_one "${HOSTS[$i]}" "${RUNS[$i]}" "${WANDB_NAMES[$i]}" "${SSL_EPOCHS[$i]}" \
        "${HEAD_TYPES[$i]}" "${EXPERIMENTS[$i]}" 1
    done
    ;;
  launch)
    for i in "${!HOSTS[@]}"; do
      launch_one "${HOSTS[$i]}" "${RUNS[$i]}" "${WANDB_NAMES[$i]}" "${SSL_EPOCHS[$i]}" \
        "${HEAD_TYPES[$i]}" "${EXPERIMENTS[$i]}" 0
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
      launch_one "${HOSTS[$i]}" "${RUNS[$i]}" "${WANDB_NAMES[$i]}" "${SSL_EPOCHS[$i]}" \
        "${HEAD_TYPES[$i]}" "${EXPERIMENTS[$i]}" 1
    done
    echo "Waiting 120s for canaries..."
    sleep 120
    health_one "${HOSTS[0]}" "${RUNS[0]}" || true
    ;;
  *)
    echo "Usage: $0 [copy|prep|canary|launch|health|all]"; exit 1
    ;;
esac
echo "Done ($MODE)."
