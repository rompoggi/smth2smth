#!/usr/bin/env bash
# Resume active fleet per report/romain.poggi/gymnote.md (W&B step = max(ckpt, log)).
# Usage: bash scripts/resume_fleet_gymnote.sh [resume|health]
set -euo pipefail

REPO="${REPO:-/Data/romain.poggi/smth2smth}"
cd "$REPO"
MODE="${1:-resume}"
EXP_DIV=track_a_diverse_arch3_divided_st_stab
EXP_Q8=track_a_diverse_arch2_perceiver_stab
EXP_Q16=track_a_diverse_arch2_perceiver_stab
SSL500="${REPO}/checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep500.pt"
RGS_PY="${REPO}/scripts/remote_fleet_rgs.py"

# host:run:kind:extra (kind=divspace -> K:seed:WID | q8 -> seed:ssl_ep:Q:WID | q16 -> seed:WID)
ENTRIES=(
  ablette:perceiverQ8-mae50-s42:q8:42:50:8:ec4la5do
  anchois:perceiverQ8-mae50-s43:q8:43:50:8:kn8m0eec
  anguille:perceiverQ8-mae50-s44:q8:44:50:8:k7j8mp3f
  barbeau:perceiverQ8-mae150-s42:q8:42:150:8:wj8plqqd
  barbue:perceiverQ8-mae150-s43:q8:43:150:8:geuyxsxx
  carrelet:perceiverQ8-mae150-s44:q8:44:150:8:h236oaie
  gardon:perceiverQ8-mae250-s42:q8:42:250:8:06opirr2
  labre:perceiverQ8-mae250-s43:q8:43:250:8:wnhdefv2
  lotte:perceiverQ8-mae250-s44:q8:44:250:8:rg7vwobj
  mulet:perceiverQ8-mae350-s42:q8:42:350:8:aphf8m7s
  murene:perceiverQ8-mae350-s43:q8:43:350:8:9byqdidr
  piranha:perceiverQ8-mae350-s44:q8:44:350:8:43oyc0ab
  raie:perceiverQ8-mae450-s42:q8:42:450:8:j28do9jj
  requin:perceiverQ8-mae450-s43:q8:43:450:8:u4yjevti
  rouget:perceiverQ8-mae450-s44:q8:44:450:8:cpfqyts2
  saumon:perceiverQ8-mae100-s44:q8:44:100:8:zybojpeg
  silure:perceiverQ8-mae200-s44:q8:44:200:8:q34m4cmr
  sole:perceiverQ8-mae300-s44:q8:44:300:8:a7rk10f0
  thon:perceiverQ8-mae400-s44:q8:44:400:8:45j0bid4
  truite:perceiverQ8-mae500-s44:q8:44:500:8:icau6rug
  lieu:perceiverQ16-mae500-s43:q16:43:6idgo6j5
  brochet:perceiverQ16-mae500-s44:q16:44:24fxahew
  requin:DivSpaceTimeK6-mae500-s44:divspace:6:44:01gnvhqm
  roussette:DivSpaceTimeK9-mae500-s44:divspace:9:44:533jwygf
)

fleet_rgs() {
  local H="$1" RUN="$2" KIND="$3"
  ssh -o BatchMode=yes "$H" "$REPO/.venv/bin/python" "$RGS_PY" "$RUN" "$KIND"
}

host_running() {
  local H="$1" RUN="$2"
  ssh -o BatchMode=yes "$H" "pgrep -af 'training.wandb.name=${RUN}' | grep -q python"
}

log_stale_sec() {
  local H="$1" RUN="$2"
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$RUN" <<'STALE'
set -euo pipefail
REPO="$1" RUN="$2"
LOG=$(ls -t "${REPO}/logs/track_a"/*/"${RUN}"_*.log "${REPO}/logs/track_a/${RUN}"_*.log 2>/dev/null | head -1 || true)
[[ -n "$LOG" && -f "$LOG" ]] || { echo 999999; exit 0; }
now=$(date +%s)
mt=$(stat -c %Y "$LOG")
echo $((now - mt))
STALE
}

resume_divspace() {
  local H="$1" RUN="$2" K="$3" SEED="$4" WID="$5" RGS="$6"
  local SEED_DIR="divspace_s${SEED}"
  local CKPT_DIR="${REPO}/checkpoints/track_a/${SEED_DIR}"
  local LOG_DIR="${REPO}/logs/track_a/${SEED_DIR}"
  local RESUME_PT="${CKPT_DIR}/${RUN}.last.pt"
  echo "=== RESUME $H $RUN rgs=$RGS wandb=$WID ==="
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$RUN" "$K" "$SEED" "$WID" "$EXP_DIV" "$SSL500" "$CKPT_DIR" "$LOG_DIR" "$RESUME_PT" "$RGS" <<'REMOTE'
set -euo pipefail
REPO="$1" RUN="$2" K="$3" SEED="$4" WID="$5" EXP="$6" SSL="$7" CKPT_DIR="$8" LOG_DIR="$9" RESUME_PT="${10}" RGS="${11}"
cd "$REPO"
mkdir -p "$LOG_DIR" "$CKPT_DIR"
test -f "$RESUME_PT" && test -f "$SSL"
LOG=$(ls -t "${LOG_DIR}/${RUN}"_*.log 2>/dev/null | head -1)
[[ -n "$LOG" ]] || LOG="${LOG_DIR}/${RUN}_$(date +%Y%m%d).log"
if pgrep -af "training.wandb.name=${RUN}" | grep -q python; then
  echo SKIP_running
  exit 0
fi
pkill -f "training.wandb.name=${RUN}" 2>/dev/null || true
sleep 2
{
  echo ""
  echo "# === RESUME $(date -Is) rgs=${RGS} wandb=${WID} ==="
} >>"$LOG"
nohup env PYTHONPATH=src PYTHONUNBUFFERED=1 WANDB_MODE=online \
  WANDB_RESUME=allow "WANDB_RUN_ID=${WID}" \
  .venv/bin/python -u -m smth2smth.pipelines.train \
  track=a "experiment=${EXP}" "seed=${SEED}" \
  "model.init_from=${SSL}" "model.temporal_layers=${K}" model.tube_t=1 dataset.num_frames=4 \
  dataset.include_val_in_train=false dataset.official_val_holdout_ratio=0 \
  "training.checkpoint_path=${CKPT_DIR}/${RUN}.pt" \
  "training.resume_from=${RESUME_PT}" \
  "training.resume_global_step=${RGS}" \
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

resume_q8() {
  local H="$1" RUN="$2" SEED="$3" SSL_EP="$4" Q="$5" WID="$6" RGS="$7"
  local SSL="${REPO}/checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep${SSL_EP}.pt"
  local RESUME_PT="${REPO}/checkpoints/track_a/videomaev2+ft/${RUN}.last.pt"
  echo "=== RESUME $H $RUN rgs=$RGS wandb=$WID ==="
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$RUN" "$SEED" "$SSL" "$Q" "$WID" "$EXP_Q8" "$RESUME_PT" "$RGS" "$SSL_EP" <<'REMOTE'
set -euo pipefail
REPO="$1" RUN="$2" SEED="$3" SSL="$4" Q="$5" WID="$6" EXP="$7" RESUME_PT="$8" RGS="$9" SSL_EP="${10}"
cd "$REPO"
mkdir -p logs/track_a checkpoints/track_a/videomaev2+ft checkpoints/track_a/ssl/pretrain
test -f "$RESUME_PT" && test -f "$SSL"
LOG=$(ls -t "logs/track_a/${RUN}"_*.log 2>/dev/null | head -1)
[[ -n "$LOG" ]] || LOG="logs/track_a/${RUN}_$(date +%Y%m%d).log"
if pgrep -af "training.wandb.name=${RUN}" | grep -q python; then
  echo SKIP_running
  exit 0
fi
pkill -f "training.wandb.name=${RUN}" 2>/dev/null || true
sleep 2
{
  echo ""
  echo "# === RESUME $(date -Is) rgs=${RGS} wandb=${WID} ==="
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

resume_q16() {
  local H="$1" RUN="$2" SEED="$3" WID="$4" RGS="$5"
  local RESUME_PT="${REPO}/checkpoints/track_a/videomaev2+ft/${RUN}.last.pt"
  echo "=== RESUME $H $RUN rgs=$RGS wandb=$WID ==="
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$RUN" "$SEED" "$WID" "$EXP_Q16" "$SSL500" "$RESUME_PT" "$RGS" <<'REMOTE'
set -euo pipefail
REPO="$1" RUN="$2" SEED="$3" WID="$4" EXP="$5" SSL="$6" RESUME_PT="$7" RGS="$8"
cd "$REPO"
mkdir -p logs/track_a checkpoints/track_a/videomaev2+ft checkpoints/track_a/ssl/pretrain
test -f "$SSL"
LOG=$(ls -t "logs/track_a/${RUN}"_*.log 2>/dev/null | head -1)
[[ -n "$LOG" ]] || LOG="logs/track_a/${RUN}_$(date +%Y%m%d).log"
if pgrep -af "training.wandb.name=${RUN}" | grep -q python; then
  echo SKIP_running
  exit 0
fi
pkill -f "training.wandb.name=${RUN}" 2>/dev/null || true
sleep 2
{
  echo ""
  echo "# === RESUME $(date -Is) rgs=${RGS} wandb=${WID} ==="
} >>"$LOG"
EXTRA_RESUME=()
[[ -f "$RESUME_PT" ]] && EXTRA_RESUME=(
  "training.resume_from=${RESUME_PT}"
  "training.resume_global_step=${RGS}"
)
nohup env PYTHONPATH=src PYTHONUNBUFFERED=1 WANDB_MODE=online \
  WANDB_RESUME=allow "WANDB_RUN_ID=${WID}" \
  .venv/bin/python -u -m smth2smth.pipelines.train \
  track=a "experiment=${EXP}" "seed=${SEED}" \
  "model.init_from=${SSL}" model.head_queries=16 model.tube_t=1 dataset.num_frames=4 \
  dataset.include_val_in_train=false dataset.official_val_holdout_ratio=0 \
  "training.checkpoint_path=checkpoints/track_a/videomaev2+ft/${RUN}.pt" \
  "${EXTRA_RESUME[@]}" \
  "training.wandb.project=smth2smth-frame-ablation" \
  ++training.wandb.group=ft-mae-scaling \
  "training.wandb.name=${RUN}" \
  ++training.wandb.config.head_type=perceiverQ16 \
  ++training.wandb.config.pretrain_epochs=500 \
  ++training.wandb.config.num_queries=16 \
  ++training.wandb.config.recipe=stab \
  >>"$LOG" 2>&1 &
echo "trainer_pid=$! log=$LOG"
REMOTE
}

do_resume() {
  # Sync helper script to all hosts
  for H in ablette anchois anguille barbeau barbue carrelet gardon labre lotte mulet murene \
    piranha raie requin rouget saumon silure sole thon truite lieu brochet roussette; do
    rsync -az "$RGS_PY" "${H}:${RGS_PY}" 2>/dev/null || true
  done

  local ok=0 skip=0 fail=0
  for e in "${ENTRIES[@]}"; do
    IFS=: read -r H RUN KIND EXTRA <<<"$e"
    ST=$(fleet_rgs "$H" "$RUN" "$KIND" 2>/dev/null) || { echo "FAIL state $H $RUN"; fail=$((fail+1)); continue; }
    RGS=$(echo "$ST" | python3 -c "import sys,json; print(json.load(sys.stdin)['rgs'])")
    EP=$(echo "$ST" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('ep') or 0)")
    DONE=$(echo "$ST" | python3 -c "import sys,json; print(json.load(sys.stdin)['done'])")
    HAS=$(echo "$ST" | python3 -c "import sys,json; print(json.load(sys.stdin)['has_ckpt'])")
    if [[ "$DONE" == "True" ]] || [[ "$EP" -ge 50 && "$KIND" != "q16" ]]; then
      echo "SKIP_DONE $H $RUN ep=$EP"
      skip=$((skip+1))
      continue
    fi
    if [[ "$HAS" != "True" && "$KIND" == "q16" ]]; then
      # fresh Q16 without ckpt yet
      IFS=: read -r SEED WID <<<"$EXTRA"
      resume_q16 "$H" "$RUN" "$SEED" "$WID" "0" && ok=$((ok+1)) || fail=$((fail+1))
      continue
    fi
    if [[ "$HAS" != "True" ]]; then
      echo "SKIP_NOCKPT $H $RUN"
      skip=$((skip+1))
      continue
    fi
    if host_running "$H" "$RUN"; then
      stale=$(log_stale_sec "$H" "$RUN")
      if [[ "$stale" -lt 900 ]]; then
        echo "SKIP_RUNNING $H $RUN stale=${stale}s"
        skip=$((skip+1))
        continue
      fi
      echo "STALE_RESTART $H $RUN stale=${stale}s"
      ssh -o BatchMode=yes "$H" "pkill -f 'training.wandb.name=${RUN}'" 2>/dev/null || true
      sleep 3
    fi
    case "$KIND" in
      divspace)
        IFS=: read -r K SEED WID <<<"$EXTRA"
        resume_divspace "$H" "$RUN" "$K" "$SEED" "$WID" "$RGS" && ok=$((ok+1)) || fail=$((fail+1))
        ;;
      q8)
        IFS=: read -r SEED SSL_EP Q WID <<<"$EXTRA"
        resume_q8 "$H" "$RUN" "$SEED" "$SSL_EP" "$Q" "$WID" "$RGS" && ok=$((ok+1)) || fail=$((fail+1))
        ;;
      q16)
        IFS=: read -r SEED WID <<<"$EXTRA"
        resume_q16 "$H" "$RUN" "$SEED" "$WID" "$RGS" && ok=$((ok+1)) || fail=$((fail+1))
        ;;
    esac
  done
  echo "resume: ok=$ok skip=$skip fail=$fail"
}

do_health() {
  sleep 90
  local ok=0
  for e in "${ENTRIES[@]}"; do
    IFS=: read -r H RUN _K _E <<<"$e"
    echo -n "$H $RUN: "
    if ssh -o BatchMode=yes "$H" "pgrep -f 'training.wandb.name=${RUN}'" >/dev/null 2>&1; then
      echo RUNNING
      ok=$((ok+1))
    else
      echo idle
    fi
  done
  echo "health running=$ok / ${#ENTRIES[@]}"
}

case "$MODE" in
  resume) do_resume ;;
  health) do_health ;;
  all) do_resume; do_health ;;
  *) echo "Usage: $0 [resume|health|all]"; exit 1 ;;
esac
