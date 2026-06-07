#!/usr/bin/env bash
# Resume/launch 11 unfinished stab-scaling runs on free fleet hosts.
# Usage: bash scripts/resume_unfinished_fleet_20260606.sh [launch|health]
set -euo pipefail

REPO="${REPO:-/Data/romain.poggi/smth2smth}"
cd "$REPO"
TAG="${TAG:-$(date +%Y%m%d)}"
MODE="${1:-launch}"
EXP_STAB=track_a_diverse_arch2_perceiver_stab
EXP_BASE=track_a_diverse_arch2_perceiver
RGS_PY="${REPO}/scripts/remote_fleet_rgs.py"
EXP_DOC=experiments/diverse_classifier_heads_post_mae.md

# host:run:mode:ssl_ep:seed:Q:WID:ckpt_src_host (mode=fresh|resume_q16_base|resume_q8_stab)
ENTRIES=(
  ablette:perceiverQ16-mae50-s42:fresh:50:42:16:::
  anguille:perceiverQ16-mae50-s43:fresh:50:43:16:::
  barbeau:perceiverQ16-mae50-s44:fresh:50:44:16:::
  barbue:perceiverQ16-mae100-s42:fresh:100:42:16:::
  baudroie:perceiverQ16-mae200-s42:fresh:200:42:16:::
  carrelet:perceiverQ16-mae300-s42:fresh:300:42:16:::
  gardon:perceiverQ16-mae400-s42:fresh:400:42:16:::
  labre:perceiverQ16-mae350-s42:resume_q16_base:350:42:16:scxr4ydc:saumon
  mulet:perceiverQ16-mae450-s43:resume_q16_base:450:43:16:npvdiorm:mulet
  murene:perceiverQ16-mae450-s44:resume_q16_base:450:44:16:2y4dnv7t:murene
  truite:perceiverQ8-mae450-s44:resume_q8_stab:450:44:8:cpfqyts2:rouget
)

copy_ssl() {
  local H="$1" EP="$2"
  local rel="checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep${EP}.pt"
  local dest="${REPO}/${rel}"
  test -f "$dest" || { echo "MISSING $dest" >&2; return 1; }
  ssh -o BatchMode=yes "$H" "mkdir -p ${REPO}/checkpoints/track_a/ssl/pretrain"
  rsync -az "$dest" "${H}:${dest}"
}

sync_ckpt_q16_base() {
  local DST_H="$1" RUN="$2" SRC_H="$3"
  local CKPT_DIR="${REPO}/checkpoints/track_a/videomaev2+ft"
  echo "=== CKPT $SRC_H -> $DST_H ($RUN) ==="
  ssh -o BatchMode=yes "$DST_H" bash -s -- "$SRC_H" "$CKPT_DIR" "$RUN" <<'RSYNC'
set -euo pipefail
SRC_H="$1" CKPT_DIR="$2" RUN="$3"
mkdir -p "$CKPT_DIR"
for suf in "" ".last"; do
  rsync -az "${SRC_H}:${CKPT_DIR}/${RUN}-NoStab${suf}.pt" "${CKPT_DIR}/${RUN}${suf}.pt"
done
RSYNC
}

sync_ckpt_canonical() {
  local DST_H="$1" RUN="$2" SRC_H="$3"
  local CKPT_DIR="${REPO}/checkpoints/track_a/videomaev2+ft"
  echo "=== CKPT $SRC_H -> $DST_H ($RUN) ==="
  ssh -o BatchMode=yes "$DST_H" bash -s -- "$SRC_H" "$CKPT_DIR" "$RUN" <<'RSYNC'
set -euo pipefail
SRC_H="$1" CKPT_DIR="$2" RUN="$3"
mkdir -p "$CKPT_DIR"
for suf in "" ".last"; do
  rsync -az "${SRC_H}:${CKPT_DIR}/${RUN}${suf}.pt" "${CKPT_DIR}/${RUN}${suf}.pt"
done
RSYNC
}

fleet_rgs() {
  local H="$1" RUN="$2" KIND="$3"
  ssh -o BatchMode=yes "$H" "$REPO/.venv/bin/python" "$RGS_PY" "$RUN" "$KIND"
}

launch_fresh_q16_stab() {
  local H="$1" RUN="$2" SSL_EP="$3" SEED="$4" Q="$5"
  local SSL="${REPO}/checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep${SSL_EP}.pt"
  local LOG="logs/track_a/${RUN}_${TAG}.log"
  echo "=== FRESH STAB $H -> $RUN ==="
  copy_ssl "$H" "$SSL_EP"
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$RUN" "$SSL" "$Q" "$SEED" "$LOG" "$EXP_STAB" "$EXP_DOC" "$SSL_EP" <<'LAUNCH'
set -euo pipefail
REPO="$1" RUN="$2" SSL="$3" Q="$4" SEED="$5" LOG="$6" EXP="$7" EXP_DOC="$8" SSL_EP="$9"
cd "$REPO"
mkdir -p logs/track_a checkpoints/track_a/videomaev2+ft checkpoints/track_a/ssl/pretrain
if pgrep -af "smth2smth.pipelines.train" | grep -v pgrep | grep -q .; then
  echo "ABORT: GPU busy on $(hostname -s)" >&2; exit 1
fi
test -f "$SSL"
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
  ++training.wandb.config.pretrain_epochs="${SSL_EP}" \
  ++training.wandb.config.num_queries="${Q}" \
  ++training.wandb.config.recipe=stab \
  >>"$LOG" 2>&1 &
echo "trainer_pid=$! log=$LOG"
LAUNCH
}

resume_q16_base() {
  local H="$1" RUN="$2" SSL_EP="$3" SEED="$4" Q="$5" WID="$6" RGS="$7"
  local SSL="${REPO}/checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep${SSL_EP}.pt"
  local RESUME_PT="${REPO}/checkpoints/track_a/videomaev2+ft/${RUN}.last.pt"
  echo "=== RESUME Q16 BASE $H -> $RUN rgs=$RGS wandb=$WID ==="
  copy_ssl "$H" "$SSL_EP"
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$RUN" "$SEED" "$SSL" "$Q" "$WID" "$EXP_BASE" "$RESUME_PT" "$RGS" "$SSL_EP" <<'REMOTE'
set -euo pipefail
REPO="$1" RUN="$2" SEED="$3" SSL="$4" Q="$5" WID="$6" EXP="$7" RESUME_PT="$8" RGS="$9" SSL_EP="${10}"
cd "$REPO"
mkdir -p logs/track_a checkpoints/track_a/videomaev2+ft
test -f "$RESUME_PT" && test -f "$SSL"
LOG=$(ls -t "logs/track_a/${RUN}"_*.log 2>/dev/null | head -1)
[[ -n "$LOG" ]] || LOG="logs/track_a/${RUN}_${TAG:-$(date +%Y%m%d)}.log"
if pgrep -af "training.wandb.name=${RUN}" | grep -q python; then echo SKIP_running; exit 0; fi
{
  echo ""
  echo "# === RESUME $(date -Is) rgs=${RGS} wandb=${WID} (base arch2) ==="
} >>"$LOG"
nohup env PYTHONPATH=src PYTHONUNBUFFERED=1 WANDB_MODE=online \
  WANDB_RESUME=allow "WANDB_RUN_ID=${WID}" \
  .venv/bin/python -u -m smth2smth.pipelines.train \
  track=a "experiment=${EXP}" "seed=${SEED}" \
  "model.init_from=${SSL}" "model.head_queries=${Q}" model.tube_t=1 dataset.num_frames=4 \
  dataset.include_val_in_train=false dataset.official_val_holdout_ratio=0 \
  "training.checkpoint_path=checkpoints/track_a/videomaev2+ft/${RUN}.pt" \
  "training.resume_from=${RESUME_PT}" \
  "training.resume_global_step=${RGS}" \
  "training.wandb.project=smth2smth-frame-ablation" \
  ++training.wandb.group=ft-mae-scaling \
  "training.wandb.name=${RUN}" \
  ++training.wandb.config.head_type="perceiverQ${Q}" \
  ++training.wandb.config.pretrain_epochs="${SSL_EP}" \
  ++training.wandb.config.num_queries="${Q}" \
  >>"$LOG" 2>&1 &
echo "trainer_pid=$! log=$LOG"
REMOTE
}

resume_q8_stab() {
  local H="$1" RUN="$2" SSL_EP="$3" SEED="$4" Q="$5" WID="$6" RGS="$7"
  local SSL="${REPO}/checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep${SSL_EP}.pt"
  local RESUME_PT="${REPO}/checkpoints/track_a/videomaev2+ft/${RUN}.last.pt"
  echo "=== RESUME Q8 STAB $H -> $RUN rgs=$RGS wandb=$WID ==="
  copy_ssl "$H" "$SSL_EP"
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$RUN" "$SEED" "$SSL" "$Q" "$WID" "$EXP_STAB" "$RESUME_PT" "$RGS" "$SSL_EP" <<'REMOTE'
set -euo pipefail
REPO="$1" RUN="$2" SEED="$3" SSL="$4" Q="$5" WID="$6" EXP="$7" RESUME_PT="$8" RGS="$9" SSL_EP="${10}"
cd "$REPO"
mkdir -p logs/track_a checkpoints/track_a/videomaev2+ft
test -f "$RESUME_PT" && test -f "$SSL"
LOG=$(ls -t "logs/track_a/${RUN}"_*.log 2>/dev/null | head -1)
[[ -n "$LOG" ]] || LOG="logs/track_a/${RUN}_${TAG:-$(date +%Y%m%d)}.log"
if pgrep -af "training.wandb.name=${RUN}" | grep -q python; then echo SKIP_running; exit 0; fi
{
  echo ""
  echo "# === RESUME $(date -Is) rgs=${RGS} wandb=${WID} (stab) ==="
} >>"$LOG"
nohup env PYTHONPATH=src PYTHONUNBUFFERED=1 WANDB_MODE=online \
  WANDB_RESUME=allow "WANDB_RUN_ID=${WID}" \
  .venv/bin/python -u -m smth2smth.pipelines.train \
  track=a "experiment=${EXP}" "seed=${SEED}" \
  "model.init_from=${SSL}" "model.head_queries=${Q}" model.tube_t=1 dataset.num_frames=4 \
  dataset.include_val_in_train=false dataset.official_val_holdout_ratio=0 \
  "training.checkpoint_path=checkpoints/track_a/videomaev2+ft/${RUN}.pt" \
  "training.resume_from=${RESUME_PT}" \
  "training.resume_global_step=${RGS}" \
  "training.wandb.project=smth2smth-frame-ablation" \
  ++training.wandb.group=ft-mae-scaling \
  "training.wandb.name=${RUN}" \
  ++training.wandb.config.head_type="perceiverQ${Q}" \
  ++training.wandb.config.pretrain_epochs="${SSL_EP}" \
  ++training.wandb.config.num_queries="${Q}" \
  ++training.wandb.config.recipe=stab \
  >>"$LOG" 2>&1 &
echo "trainer_pid=$! log=$LOG"
REMOTE
}

do_launch() {
  for ent in "${ENTRIES[@]}"; do
    H="${ent%%:*}"
    rsync -az "$RGS_PY" "${H}:${RGS_PY}" 2>/dev/null || true
  done
  for ent in "${ENTRIES[@]}"; do
    IFS=: read -r H RUN MODE SSL_EP SEED Q WID CKPT_SRC <<<"$ent"
    case "$MODE" in
      fresh)
        launch_fresh_q16_stab "$H" "$RUN" "$SSL_EP" "$SEED" "$Q" || echo "FAIL $H $RUN"
        ;;
      resume_q16_base)
        sync_ckpt_q16_base "$H" "$RUN" "$CKPT_SRC"
        ST=$(fleet_rgs "$H" "$RUN" q16)
        RGS=$(echo "$ST" | python3 -c "import sys,json; print(json.load(sys.stdin)['rgs'])")
        resume_q16_base "$H" "$RUN" "$SSL_EP" "$SEED" "$Q" "$WID" "$RGS" || echo "FAIL $H $RUN"
        ;;
      resume_q8_stab)
        sync_ckpt_canonical "$H" "$RUN" "$CKPT_SRC"
        ST=$(fleet_rgs "$H" "$RUN" q8)
        RGS=$(echo "$ST" | python3 -c "import sys,json; print(json.load(sys.stdin)['rgs'])")
        resume_q8_stab "$H" "$RUN" "$SSL_EP" "$SEED" "$Q" "$WID" "$RGS" || echo "FAIL $H $RUN"
        ;;
    esac
  done
}

do_health() {
  sleep 90
  local ok=0
  for ent in "${ENTRIES[@]}"; do
    IFS=: read -r H RUN _ <<<"$ent"
    echo -n "$H $RUN: "
    if ssh -o BatchMode=yes "$H" "pgrep -f 'training.wandb.name=${RUN}'" >/dev/null 2>&1; then
      echo RUNNING
      ok=$((ok + 1))
      ssh -o BatchMode=yes "$H" "LOG=\$(ls -t ${REPO}/logs/track_a/${RUN}_*.log 2>/dev/null | head -1); tail -3 \"\$LOG\" 2>/dev/null"
    else
      echo idle
    fi
  done
  echo "health running=$ok / ${#ENTRIES[@]}"
}

case "$MODE" in
  launch) do_launch ;;
  health) do_health ;;
  all) do_launch; do_health ;;
  *) echo "Usage: $0 [launch|health|all]"; exit 1 ;;
esac
