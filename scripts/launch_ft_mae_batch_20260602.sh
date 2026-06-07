#!/usr/bin/env bash
# Complete 3 incomplete ft-mae-scaling runs + DivSpaceTime K{1,3,6,9,12} seeds 43/44.
# Usage: bash scripts/launch_ft_mae_batch_20260602.sh [sync-ssl|sync-code|launch|health]
set -euo pipefail

REPO="${REPO:-/Data/romain.poggi/smth2smth}"
TAG="${TAG:-$(date +%Y%m%d)}"
MODE="${1:-launch}"
SRC_SSL="${SRC_SSL:-gymnote}"
SSL500="${REPO}/checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep500.pt"
SSL150="${REPO}/checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep150.pt"
SSL200="${REPO}/checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep200.pt"
EXP_DIV=track_a_diverse_arch3_divided_st_stab

ALL_HOSTS=(
  rouget labre anchois
  ablette anguille barbue carrelet lotte
  piranha raie requin roussette sole
)

sync_code() {
  local H="$1"
  local here
  here="$(hostname -s 2>/dev/null || hostname)"
  [[ "$H" == "$here" ]] && return 0
  rsync -az "${REPO}/src/" "${H}:${REPO}/src/"
  rsync -az "${REPO}/configs/" "${H}:${REPO}/configs/"
}

sync_ssl_one() {
  local H="$1" rel="$2"
  local src="${REPO}/${rel}"
  local here
  here="$(hostname -s 2>/dev/null || hostname)"
  if [[ "$here" != "$SRC_SSL" ]]; then
    ssh -o BatchMode=yes "$SRC_SSL" "test -f $src" || { echo "MISSING on $SRC_SSL: $src" >&2; return 1; }
  else
    test -f "$src" || { echo "MISSING locally: $src" >&2; return 1; }
  fi
  ssh -o BatchMode=yes "$H" "mkdir -p ${REPO}/checkpoints/track_a/ssl/pretrain"
  if [[ "$here" == "$SRC_SSL" ]]; then
    rsync -az "$src" "${H}:${src}"
  else
    rsync -az "${SRC_SSL}:${src}" "${H}:${src}"
  fi
}

sync_ssl_all() {
  for H in "${ALL_HOSTS[@]}"; do
    ssh -o BatchMode=yes "$H" "test -f $SSL500" 2>/dev/null || sync_ssl_one "$H" "checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep500.pt"
  done
  for H in rouget; do
    ssh -o BatchMode=yes "$H" "test -f $SSL150" 2>/dev/null || sync_ssl_one "$H" "checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep150.pt"
  done
  for H in anchois; do
    ssh -o BatchMode=yes "$H" "test -f $SSL500" 2>/dev/null || sync_ssl_one "$H" "checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep500.pt"
  done
}

prep_ckpt_perceiver_q4() {
  echo "=== copy perceiverQ4 resume ckpt rouget -> anchois (via scp on destination) ==="
  local src_last="${REPO}/checkpoints/track_a/videomaev2+ft/arch2-perceiver-q4-trainonly.last.pt"
  local src_pt="${REPO}/checkpoints/track_a/videomaev2+ft/arch2-perceiver-q4-trainonly.pt"
  local dst_last="${REPO}/checkpoints/track_a/videomaev2+ft/perceiverQ4-mae500-s42.last.pt"
  local dst_pt="${REPO}/checkpoints/track_a/videomaev2+ft/perceiverQ4-mae500-s42.pt"
  ssh -o BatchMode=yes anchois "mkdir -p ${REPO}/checkpoints/track_a/videomaev2+ft"
  ssh -o BatchMode=yes anchois \
    "scp -o BatchMode=yes rouget:${src_last} ${dst_last}"
  ssh -o BatchMode=yes anchois \
    "scp -o BatchMode=yes rouget:${src_pt} ${dst_pt} 2>/dev/null || true"
}

launch_resume_meanpool() {
  local H="$1" RUN="$2" SEED="$3" SSL_EP="$4" WID="$5" TAG_LOG="$6"
  local SSL="${REPO}/checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep${SSL_EP}.pt"
  echo "=== RESUME $H $RUN ==="
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$RUN" "$SEED" "$SSL" "$WID" "$TAG_LOG" "$SSL_EP" <<'REMOTE'
set -euo pipefail
REPO="$1" RUN="$2" SEED="$3" SSL="$4" WID="$5" TAG_LOG="$6" SSL_EP="$7"
cd "$REPO"
CKPT_DIR="${REPO}/checkpoints/track_a/videomaev2+ft"
RESUME_PT="${CKPT_DIR}/${RUN}.last.pt"
[[ -f "$RESUME_PT" ]] || RESUME_PT="${CKPT_DIR}/${RUN}.pt"
[[ -f "$RESUME_PT" ]] || { echo "ABORT: no ckpt" >&2; exit 1; }
test -f "$SSL" || { echo "ABORT: missing SSL" >&2; exit 1; }

LOG="logs/track_a/${RUN}_${TAG_LOG}.log"
if compgen -G "${REPO}/logs/track_a/${RUN}_"*.log >/dev/null; then
  LOG=$(ls -t "${REPO}/logs/track_a/${RUN}"_*.log | head -1)
fi
mkdir -p logs/track_a
if [[ ! -f "$LOG" ]]; then
  {
    echo "# run: ${RUN}"
    echo "# started: $(date -Is)"
    echo "# track: a"
    echo "# experiment_doc: experiments/experimental_cleanup.md"
    echo "# hydra: experiment=track_a_videomae_official_ssv2_ft seed=${SEED}"
  } >"$LOG"
fi

pkill -f "training.wandb.name=${RUN}" 2>/dev/null || true
sleep 2

EPOCH=$("${REPO}/.venv/bin/python" -c "import torch; c=torch.load('${RESUME_PT}',map_location='cpu',weights_only=False); print(int((c.get('extra')or{}).get('epoch',0)))")
GSTEP=$((EPOCH * 5625))

{
  echo ""
  echo "# === RESUME $(date -Is) ==="
  echo "# resume_from: ${RESUME_PT}"
  echo "# resume_epoch: ${EPOCH}"
  echo "# resume_global_step: ${GSTEP}"
  echo "# wandb_run_id: ${WID}"
} >>"$LOG"

nohup env PYTHONPATH=src PYTHONUNBUFFERED=1 WANDB_MODE=online \
  WANDB_RESUME=allow "WANDB_RUN_ID=${WID}" \
  .venv/bin/python -u -m smth2smth.pipelines.train \
  track=a experiment=track_a_videomae_official_ssv2_ft "seed=${SEED}" \
  "model.init_from=${SSL}" model.tube_t=1 dataset.num_frames=4 \
  dataset.include_val_in_train=false dataset.official_val_holdout_ratio=0 \
  "training.checkpoint_path=${CKPT_DIR}/${RUN}.pt" \
  "training.resume_from=${RESUME_PT}" \
  "training.resume_global_step=${GSTEP}" \
  "training.wandb.project=smth2smth-frame-ablation" \
  ++training.wandb.group=ft-mae-scaling \
  "training.wandb.name=${RUN}" \
  ++training.wandb.config.head_type=meanpool \
  ++training.wandb.config.pretrain_epochs="${SSL_EP}" \
  >>"$LOG" 2>&1 &
echo "trainer_pid=$! log=$LOG epoch=${EPOCH} gstep=${GSTEP}"
REMOTE
}

launch_resume_perceiver_q4() {
  echo "=== RESUME anchois perceiverQ4-mae500-s42 ==="
  ssh -o BatchMode=yes anchois bash -s -- "$REPO" "$TAG" <<'REMOTE'
set -euo pipefail
REPO="$1" TAG="$2"
cd "$REPO"
RUN=perceiverQ4-mae500-s42
WID=s42-replay-v2-perceiverQ4-mae500
CKPT_DIR="${REPO}/checkpoints/track_a/videomaev2+ft"
RESUME_PT="${CKPT_DIR}/${RUN}.last.pt"
SSL="${REPO}/checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep500.pt"
[[ -f "$RESUME_PT" ]] || { echo "ABORT: no ckpt" >&2; exit 1; }
test -f "$SSL" || { echo "ABORT: missing SSL" >&2; exit 1; }

LOG="logs/track_a/${RUN}_${TAG}.log"
for d in logs/track_a logs/track_a/perceiver_q_s42; do
  if compgen -G "${REPO}/${d}/${RUN}_"*.log >/dev/null; then
    LOG=$(ls -t "${REPO}/${d}/${RUN}"_*.log | head -1)
    break
  fi
done
mkdir -p logs/track_a
if [[ ! -f "$LOG" ]]; then
  {
    echo "# run: ${RUN}"
    echo "# started: $(date -Is)"
    echo "# track: a"
    echo "# experiment_doc: experiments/experimental_cleanup.md"
    echo "# hydra: experiment=track_a_diverse_arch2_perceiver_stab seed=42"
  } >"$LOG"
fi

pkill -f "training.wandb.name=${RUN}" 2>/dev/null || true
sleep 2

EPOCH=$("${REPO}/.venv/bin/python" -c "import torch; c=torch.load('${RESUME_PT}',map_location='cpu',weights_only=False); print(int((c.get('extra')or{}).get('epoch',0)))")
GSTEP=$((EPOCH * 5625))

{
  echo ""
  echo "# === RESUME $(date -Is) ==="
  echo "# resume_from: ${RESUME_PT}"
  echo "# resume_epoch: ${EPOCH}"
  echo "# resume_global_step: ${GSTEP}"
  echo "# wandb_run_id: ${WID}"
} >>"$LOG"

nohup env PYTHONPATH=src PYTHONUNBUFFERED=1 WANDB_MODE=online \
  WANDB_RESUME=allow "WANDB_RUN_ID=${WID}" \
  .venv/bin/python -u -m smth2smth.pipelines.train \
  track=a experiment=track_a_diverse_arch2_perceiver_stab seed=42 \
  "model.init_from=${SSL}" model.head_queries=4 model.tube_t=1 dataset.num_frames=4 \
  dataset.include_val_in_train=false dataset.official_val_holdout_ratio=0 \
  "training.checkpoint_path=${CKPT_DIR}/${RUN}.pt" \
  "training.resume_from=${RESUME_PT}" \
  "training.resume_global_step=${GSTEP}" \
  "training.wandb.project=smth2smth-frame-ablation" \
  ++training.wandb.group=ft-mae-scaling \
  "training.wandb.name=${RUN}" \
  ++training.wandb.config.head_type=perceiverQ4 \
  ++training.wandb.config.num_queries=4 \
  ++training.wandb.config.pretrain_epochs=500 \
  >>"$LOG" 2>&1 &
echo "trainer_pid=$! log=$LOG epoch=${EPOCH} gstep=${GSTEP}"
REMOTE
}

launch_divspace() {
  local H="$1" RUN="$2" K="$3" SEED="$4"
  local CKPT_DIR="${REPO}/checkpoints/track_a/divspace_s${SEED}"
  local LOG_DIR="logs/track_a/divspace_s${SEED}"
  echo "=== FRESH $H $RUN K=$K seed=$SEED ==="
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$RUN" "$K" "$SEED" "$EXP_DIV" "$SSL500" "$CKPT_DIR" "$LOG_DIR" "$TAG" <<'REMOTE'
set -euo pipefail
REPO="$1" RUN="$2" K="$3" SEED="$4" EXP="$5" SSL="$6" CKPT_DIR="$7" LOG_DIR="$8" TAG="$9"
cd "$REPO"
mkdir -p "$LOG_DIR" "$CKPT_DIR"
LOG="${LOG_DIR}/${RUN}_${TAG}.log"
test -f "$SSL" || { echo "ABORT: missing SSL" >&2; exit 1; }

pkill -f "training.wandb.name=${RUN}" 2>/dev/null || true
sleep 2
if pgrep -af "training.wandb.name=${RUN}" | grep -q python; then
  echo "ABORT: already running" >&2
  exit 1
fi

{
  echo "# run: ${RUN}"
  echo "# started: $(date -Is)"
  echo "# track: a"
  echo "# experiment_doc: experiments/diverse_classifier_heads_post_mae.md"
  echo "# hydra: experiment=${EXP} model.temporal_layers=${K} seed=${SEED}"
  echo "# host: $(hostname)"
} >>"$LOG"

nohup env PYTHONPATH=src PYTHONUNBUFFERED=1 WANDB_MODE=online \
  .venv/bin/python -u -m smth2smth.pipelines.train \
  track=a "experiment=${EXP}" "seed=${SEED}" \
  "model.init_from=${SSL}" "model.temporal_layers=${K}" model.tube_t=1 dataset.num_frames=4 \
  dataset.include_val_in_train=false dataset.official_val_holdout_ratio=0 \
  "training.checkpoint_path=${CKPT_DIR}/${RUN}.pt" \
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
  local H="$1" LOG="$2"
  echo "=== HEALTH $H ==="
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$LOG" <<'HEALTH'
set -euo pipefail
REPO="$1" LOG="$2"
cd "$REPO"
pgrep -af "smth2smth.pipelines.train" | grep python | grep -v pgrep | head -1 || { echo "FAIL: no trainer"; exit 1; }
test -f "$LOG" || { echo "FAIL: no log $LOG"; exit 1; }
tail -6 "$LOG"
grep -q Traceback "$LOG" && { echo "FAIL: traceback"; exit 1; }
grep -qE '\[.*\] step [0-9]+/' "$LOG" || grep -q 'Resuming from checkpoint' "$LOG" || { echo "FAIL: no progress yet"; exit 1; }
echo OK
HEALTH
}

case "$MODE" in
  sync-code)
    for H in "${ALL_HOSTS[@]}"; do sync_code "$H"; done
    ;;
  sync-ssl)
    sync_ssl_all
    prep_ckpt_perceiver_q4
    ;;
  resume-meanpool)
    launch_resume_meanpool rouget meanpool-mae150-s42 42 150 s42-replay-v2-mae150 20260525
    ;;
  launch)
    for H in "${ALL_HOSTS[@]}"; do sync_code "$H"; done
    sync_ssl_all
    prep_ckpt_perceiver_q4

    launch_resume_meanpool rouget meanpool-mae150-s42 42 150 s42-replay-v2-mae150 20260525
    launch_resume_meanpool labre meanpool-mae200-s44 44 200 ia58rbey 20260531
    launch_resume_perceiver_q4

    launch_divspace ablette DivSpaceTimeK1-mae500-s43 1 43
    launch_divspace anguille DivSpaceTimeK3-mae500-s43 3 43
    launch_divspace barbue DivSpaceTimeK6-mae500-s43 6 43
    launch_divspace carrelet DivSpaceTimeK9-mae500-s43 9 43
    launch_divspace lotte DivSpaceTimeK12-mae500-s43 12 43

    launch_divspace piranha DivSpaceTimeK1-mae500-s44 1 44
    launch_divspace raie DivSpaceTimeK3-mae500-s44 3 44
    launch_divspace requin DivSpaceTimeK6-mae500-s44 6 44
    launch_divspace roussette DivSpaceTimeK9-mae500-s44 9 44
    launch_divspace sole DivSpaceTimeK12-mae500-s44 12 44

    echo "Waiting 90s for health..."
    sleep 90
  ;;
  health)
    health_one rouget "$(ssh rouget 'ls -t /Data/romain.poggi/smth2smth/logs/track_a/meanpool-mae150-s42_*.log | head -1')" || true
    health_one labre "$(ssh labre 'ls -t /Data/romain.poggi/smth2smth/logs/track_a/meanpool-mae200-s44_*.log | head -1')" || true
    health_one anchois "$(ssh anchois 'ls -t /Data/romain.poggi/smth2smth/logs/track_a/perceiverQ4-mae500-s42_*.log 2>/dev/null | head -1')" || true
    health_one ablette "logs/track_a/divspace_s43/DivSpaceTimeK1-mae500-s43_${TAG}.log" || true
    health_one anguille "logs/track_a/divspace_s43/DivSpaceTimeK3-mae500-s43_${TAG}.log" || true
    health_one barbue "logs/track_a/divspace_s43/DivSpaceTimeK6-mae500-s43_${TAG}.log" || true
    health_one carrelet "logs/track_a/divspace_s43/DivSpaceTimeK9-mae500-s43_${TAG}.log" || true
    health_one lotte "logs/track_a/divspace_s43/DivSpaceTimeK12-mae500-s43_${TAG}.log" || true
    health_one piranha "logs/track_a/divspace_s44/DivSpaceTimeK1-mae500-s44_${TAG}.log" || true
    health_one raie "logs/track_a/divspace_s44/DivSpaceTimeK3-mae500-s44_${TAG}.log" || true
    health_one requin "logs/track_a/divspace_s44/DivSpaceTimeK6-mae500-s44_${TAG}.log" || true
    health_one roussette "logs/track_a/divspace_s44/DivSpaceTimeK9-mae500-s44_${TAG}.log" || true
    health_one sole "logs/track_a/divspace_s44/DivSpaceTimeK12-mae500-s44_${TAG}.log" || true
    ;;
  *)
    echo "Usage: $0 [sync-ssl|sync-code|launch|health]"; exit 1 ;;
esac
echo "Done ($MODE)."
