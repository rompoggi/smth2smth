#!/usr/bin/env bash
# Launch Perceiver Q8 stab SSL grid (20 runs on 20 free hosts).
# Usage: bash scripts/launch_q8_stab_ssl_grid.sh
set -euo pipefail

REPO="${REPO:-/Data/romain.poggi/smth2smth}"
cd "$REPO"
TAG="${TAG:-$(date +%Y%m%d)}"
EXPERIMENT_STAB=track_a_diverse_arch2_perceiver_stab
EXP_DOC=experiments/diverse_classifier_heads_post_mae.md

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
  echo "=== LAUNCH $H -> $RUN (SSL ep${SSL_EP} seed${SEED}) ==="
  copy_ssl "$H" "$SSL_EP"
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$RUN" "$SSL" "$Q" "$SEED" "$LOG" "$EXPERIMENT_STAB" "$EXP_DOC" <<'LAUNCH'
set -euo pipefail
REPO="$1" RUN="$2" SSL="$3" Q="$4" SEED="$5" LOG="$6" EXP="$7" EXP_DOC="$8"
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
  echo "# experiment_doc: ${EXP_DOC}"
  echo "# hydra: experiment=${EXP} seed=${SEED} model.head_queries=${Q}"
  echo "# recipe: stab"
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

# host:run:ssl_ep:seed (Q=8)
ENTRIES=(
  ablette:perceiverQ8-mae50-s42:50:42
  anchois:perceiverQ8-mae50-s43:50:43
  anguille:perceiverQ8-mae50-s44:50:44
  barbeau:perceiverQ8-mae150-s42:150:42
  barbue:perceiverQ8-mae150-s43:150:43
  carrelet:perceiverQ8-mae150-s44:150:44
  gardon:perceiverQ8-mae250-s42:250:42
  labre:perceiverQ8-mae250-s43:250:43
  lotte:perceiverQ8-mae250-s44:250:44
  mulet:perceiverQ8-mae350-s42:350:42
  murene:perceiverQ8-mae350-s43:350:43
  piranha:perceiverQ8-mae350-s44:350:44
  raie:perceiverQ8-mae450-s42:450:42
  requin:perceiverQ8-mae450-s43:450:43
  rouget:perceiverQ8-mae450-s44:450:44
  saumon:perceiverQ8-mae100-s44:100:44
  silure:perceiverQ8-mae200-s44:200:44
  sole:perceiverQ8-mae300-s44:300:44
  thon:perceiverQ8-mae400-s44:400:44
  truite:perceiverQ8-mae500-s44:500:44
)

for ent in "${ENTRIES[@]}"; do
  IFS=: read -r H RUN EP SEED <<<"$ent"
  launch_perceiver_stab "$H" "$RUN" "$EP" 8 "$SEED" &
done
wait
echo "All ${#ENTRIES[@]} launches submitted."
