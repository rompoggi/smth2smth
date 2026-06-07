#!/usr/bin/env bash
# Fleet reorg 2026-06-02: pause non-stab Q16 mae450, stop K12, rename NoStab, launch Q8 stab + queue Q16 mae500 stab.
# Usage: bash scripts/fleet_stab_reorg_20260602.sh [stop|rename|launch-q8|queue-q16|all]
set -euo pipefail

REPO="${REPO:-/Data/romain.poggi/smth2smth}"
cd "$REPO"
TAG="${TAG:-$(date +%Y%m%d)}"
EXPERIMENT_STAB=track_a_diverse_arch2_perceiver_stab
MODE="${1:-all}"

stop_run_on_host() {
  local H="$1" RUN="$2" REASON="$3" LOG_GLOB="${4:-}"
  echo "=== STOP $RUN on $H ($REASON) ==="
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$RUN" "$REASON" "$LOG_GLOB" <<'STOP'
set -euo pipefail
REPO="$1" RUN="$2" REASON="$3" LOG_GLOB="$4"
pkill -f "training.wandb.name=${RUN}" 2>/dev/null || true
sleep 3
if pgrep -f "training.wandb.name=${RUN}" >/dev/null 2>&1; then
  echo "FAIL: ${RUN} still running on $(hostname -s)" >&2
  exit 1
fi
LOG=""
if [[ -n "$LOG_GLOB" ]]; then
  LOG=$(ls -t ${REPO}/${LOG_GLOB} 2>/dev/null | head -1 || true)
fi
if [[ -z "$LOG" ]]; then
  LOG=$(ls -t "$REPO/logs/track_a/${RUN}"_*.log 2>/dev/null | head -1 || true)
fi
if [[ -n "$LOG" && -f "$LOG" ]]; then
  {
    echo ""
    echo "# === STOPPED $(date -Is): ${REASON} ==="
  } >>"$LOG"
fi
echo "stopped ${RUN} on $(hostname -s)"
STOP
}

rename_ckpt_on_host() {
  local H="$1" OLD="$2"
  local NEW="${OLD}-NoStab"
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$OLD" "$NEW" <<'REN'
set -euo pipefail
REPO="$1" OLD="$2" NEW="$3"
for dir in "$REPO/checkpoints/track_a/videomaev2+ft" \
           "$REPO/checkpoints/track_a/divspace_s42" \
           "$REPO/checkpoints/track_a/divspace_s43" \
           "$REPO/checkpoints/track_a/divspace_s44"; do
  for ext in .pt .last.pt; do
    src="${dir}/${OLD}${ext}"
    dst="${dir}/${NEW}${ext}"
    if [[ -f "$src" && ! -f "$dst" ]]; then
      mv "$src" "$dst"
      echo "renamed $(hostname -s): $src -> $dst"
    fi
  done
done
REN
}

# All fleet hosts that may hold perceiverQ16 scaling ckpts.
RENAME_HOSTS=(
  gardon gymnote labre lieu lotte mulet murene piranha raie requin rouget roussette sole thon truite
  ablette anguille barbeau barbue baudroie carrelet saumon silure anchois brochet
)

do_stop() {
  stop_run_on_host mulet perceiverQ16-mae450-s43 "PAUSED not stab config" \
    "logs/track_a/perceiverQ16-mae450-s43_*.log"
  stop_run_on_host murene perceiverQ16-mae450-s44 "PAUSED not stab config" \
    "logs/track_a/perceiverQ16-mae450-s44_*.log"
  stop_run_on_host thon DivSpaceTimeK12-mae500-s42 "STOPPED K12 too slow" \
    "logs/track_a/divspace_s42/DivSpaceTimeK12-mae500-s42_*.log"
  stop_run_on_host lotte DivSpaceTimeK12-mae500-s43 "STOPPED K12 too slow" \
    "logs/track_a/divspace_s43/DivSpaceTimeK12-mae500-s43_*.log"
  stop_run_on_host sole DivSpaceTimeK12-mae500-s44 "STOPPED K12 too slow" \
    "logs/track_a/divspace_s44/DivSpaceTimeK12-mae500-s44_*.log"
}

do_rename() {
  echo "=== W&B rename non-stab PerceiverQ* -> *-NoStab ==="
  PYTHONPATH=src uv run python scripts/rename_perceiver_nostab_wandb.py
  echo "=== Local checkpoint rename on fleet ==="
  while IFS=$'\t' read -r old new _rid; do
    [[ "$old" == "#"* || -z "$old" ]] && continue
    for H in "${RENAME_HOSTS[@]}"; do
      rename_ckpt_on_host "$H" "$old" || true
    done
  done <"$REPO/logs/track_a/perceiver_nostab_rename_manifest.txt"
}

copy_ssl() {
  local H="$1" EP="$2"
  local rel="checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep${EP}.pt"
  local dest="${REPO}/${rel}"
  test -f "$dest" || { echo "MISSING $dest" >&2; return 1; }
  ssh -o BatchMode=yes "$H" "mkdir -p ${REPO}/checkpoints/track_a/ssl/pretrain"
  rsync -az "$dest" "${H}:${dest}"
}

launch_perceiver_stab() {
  local H="$1" RUN="$2" SSL_EP="$3" Q="$4" SEED="$5"
  local SSL="${REPO}/checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep${SSL_EP}.pt"
  local LOG="logs/track_a/${RUN}_${TAG}.log"
  echo "=== LAUNCH STAB $H -> $RUN Q=$Q seed=$SEED SSL ep$SSL_EP ==="
  copy_ssl "$H" "$SSL_EP"
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$RUN" "$SSL" "$Q" "$SEED" "$LOG" "$EXPERIMENT_STAB" <<'LAUNCH'
set -euo pipefail
REPO="$1" RUN="$2" SSL="$3" Q="$4" SEED="$5" LOG="$6" EXP="$7"
cd "$REPO"
mkdir -p logs/track_a checkpoints/track_a/videomaev2+ft checkpoints/track_a/ssl/pretrain
if pgrep -af "smth2smth.pipelines.train" | grep -v pgrep | grep -q .; then
  echo "ABORT: GPU busy on $(hostname -s)" >&2
  pgrep -af "smth2smth.pipelines.train" | grep -v pgrep || true
  exit 1
fi
test -f "$SSL" || { echo "missing SSL $SSL" >&2; exit 1; }
: >"$LOG"
{
  echo "# run: ${RUN}"
  echo "# started: $(date -Is)"
  echo "# track: a"
  echo "# experiment_doc: experiments/diverse_classifier_heads_post_mae.md"
  echo "# hydra: experiment=${EXP} seed=${SEED} model.head_queries=${Q} T=4 train-only SSL ep from init"
  echo "# recipe: stab (new_module_lr=1e-4, holdout_ratio=0)"
  echo "# host: $(hostname)"
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
  ++training.wandb.config.head_type="perceiverQ${Q}" \
  ++training.wandb.config.pretrain_epochs="$(basename "$SSL" | sed -n 's/.*ep\([0-9]*\)\.pt/\1/p')" \
  ++training.wandb.config.num_queries="${Q}" \
  ++training.wandb.config.recipe=stab \
  >>"$LOG" 2>&1 &
echo "trainer_pid=$! log=$LOG"
LAUNCH
}

do_launch_q8() {
  # 8 runs: Q8 @ SSL {100,200,300,400} x seeds {42,43}
  launch_perceiver_stab anchois perceiverQ8-mae100-s42 100 8 42
  launch_perceiver_stab labre  perceiverQ8-mae100-s43 100 8 43
  launch_perceiver_stab truite  perceiverQ8-mae200-s42 200 8 42
  launch_perceiver_stab thon    perceiverQ8-mae200-s43 200 8 43
  launch_perceiver_stab lotte   perceiverQ8-mae300-s42 300 8 42
  launch_perceiver_stab sole    perceiverQ8-mae300-s43 300 8 43
  launch_perceiver_stab mulet   perceiverQ8-mae400-s42 400 8 42
  launch_perceiver_stab murene  perceiverQ8-mae400-s43 400 8 43
}

do_queue_q16() {
  local CHAIN_LOG="logs/track_a/chain_q16_mae500_stab_${TAG}.log"
  mkdir -p logs/track_a
  {
    echo "# chain: Q16 mae500 stab after DivSpaceTime finishes"
    echo "# started: $(date -Is)"
  } >"$CHAIN_LOG"
  nohup env REPO="$REPO" TAG="$TAG" EXPERIMENT_STAB="$EXPERIMENT_STAB" bash -s >>"$CHAIN_LOG" 2>&1 <<'CHAIN' &
set -euo pipefail
wait_remote() {
  local H="$1" RUN="$2"
  echo "[$(date -Is)] waiting for ${RUN} on ${H}..."
  while ssh -o BatchMode=yes "$H" "pgrep -f 'training.wandb.name=${RUN}'" >/dev/null 2>&1; do
    sleep 120
  done
  echo "[$(date -Is)] ${RUN} finished on ${H}"
}
launch_remote() {
  local H="$1" RUN="$2" SSL_EP="$3" Q="$4" SEED="$5"
  local SSL="${REPO}/checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep${SSL_EP}.pt"
  local LOG="logs/track_a/${RUN}_${TAG}.log"
  echo "[$(date -Is)] launching stab ${RUN} on ${H}"
  ssh -o BatchMode=yes "$H" "mkdir -p ${REPO}/checkpoints/track_a/ssl/pretrain"
  rsync -az "$SSL" "${H}:${SSL}" 2>/dev/null || true
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$RUN" "$SSL" "$Q" "$SEED" "$LOG" "$EXPERIMENT_STAB" "$SSL_EP" <<'LAUNCH'
set -euo pipefail
REPO="$1" RUN="$2" SSL="$3" Q="$4" SEED="$5" LOG="$6" EXP="$7" SSL_EP="$8"
cd "$REPO"
mkdir -p logs/track_a checkpoints/track_a/videomaev2+ft
if pgrep -af "smth2smth.pipelines.train" | grep -v pgrep | grep -q .; then
  echo "ABORT: GPU busy" >&2; exit 1
fi
: >"$LOG"
{
  echo "# run: ${RUN}"
  echo "# started: $(date -Is)"
  echo "# track: a"
  echo "# experiment_doc: experiments/diverse_classifier_heads_post_mae.md"
  echo "# hydra: experiment=${EXP} seed=${SEED} model.head_queries=${Q} T=4 train-only"
  echo "# recipe: stab (replaces prior *-NoStab scaling run)"
  echo "# queued_after: DivSpaceTime on $(hostname)"
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
  ++training.wandb.config.head_type="perceiverQ${Q}" \
  ++training.wandb.config.pretrain_epochs="${SSL_EP}" \
  ++training.wandb.config.num_queries="${Q}" \
  ++training.wandb.config.recipe=stab \
  >>"$LOG" 2>&1 &
echo "trainer_pid=$! log=$LOG"
LAUNCH
}
wait_remote lieu DivSpaceTimeK9-mae500-s42
launch_remote lieu perceiverQ16-mae500-s43 500 16 43
wait_remote brochet DivSpaceTimeK6-mae500-s42
launch_remote brochet perceiverQ16-mae500-s44 500 16 44
echo "[$(date -Is)] Q16 mae500 stab chain done"
CHAIN
  echo "chain_pid=$! log=$CHAIN_LOG"
}

case "$MODE" in
  stop) do_stop ;;
  rename) do_rename ;;
  launch-q8) do_launch_q8 ;;
  queue-q16) do_queue_q16 ;;
  all)
    do_stop
    do_rename
    do_launch_q8
    do_queue_q16
    echo "Waiting 90s for Q8 boot..."
    sleep 90
    ssh -o BatchMode=yes anchois "tail -20 ${REPO}/logs/track_a/perceiverQ8-mae100-s42_${TAG}.log" || true
    ;;
  *) echo "Usage: $0 [stop|rename|launch-q8|queue-q16|all]"; exit 1 ;;
esac
echo "Done ($MODE)."
