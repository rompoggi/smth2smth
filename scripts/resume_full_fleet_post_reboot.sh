#!/usr/bin/env bash
# Resume full active fleet after GPU restart (DivSpaceTime + Q8 stab + rouget meanpool + Q16 chain).
# Skips: K12, *-NoStab, paused m450.
# Usage: bash scripts/resume_full_fleet_post_reboot.sh [all|resume|chain-q16|health]
set -euo pipefail

REPO="${REPO:-/Data/romain.poggi/smth2smth}"
cd "$REPO"
TAG="${TAG:-$(date +%Y%m%d)}"
MODE="${1:-all}"
EXP_STAB=track_a_diverse_arch2_perceiver_stab

ckpt_global_step() {
  local H="$1" CKPT="$2"
  ssh -o BatchMode=yes "$H" "$REPO/.venv/bin/python" - "$CKPT" <<'PY'
import sys, torch
p = sys.argv[1]
c = torch.load(p, map_location="cpu", weights_only=False)
gs = (c.get("extra") or {}).get("global_step")
print(int(gs) if gs is not None else 0)
PY
}

resume_q8_stab() {
  local H="$1" RUN="$2" SEED="$3" SSL_EP="$4" Q="$5" WID="$6"
  local SSL="${REPO}/checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep${SSL_EP}.pt"
  local CKPT_DIR="${REPO}/checkpoints/track_a/videomaev2+ft"
  local RESUME_PT="${CKPT_DIR}/${RUN}.last.pt"
  local GSTEP
  GSTEP=$(ckpt_global_step "$H" "$RESUME_PT") || { echo "FAIL: no ckpt $H $RUN"; return 1; }
  echo "=== RESUME Q8 STAB $H $RUN gs=$GSTEP wandb=$WID ==="
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$RUN" "$SEED" "$SSL" "$Q" "$WID" "$EXP_STAB" "$RESUME_PT" "$GSTEP" <<'REMOTE'
set -euo pipefail
REPO="$1" RUN="$2" SEED="$3" SSL="$4" Q="$5" WID="$6" EXP="$7" RESUME_PT="$8" GSTEP="$9"
cd "$REPO"
mkdir -p logs/track_a checkpoints/track_a/videomaev2+ft
test -f "$RESUME_PT" || { echo "ABORT: missing $RESUME_PT" >&2; exit 1; }
test -f "$SSL" || { echo "ABORT: missing SSL" >&2; exit 1; }
LOG=$(ls -t "logs/track_a/${RUN}"_*.log 2>/dev/null | head -1)
[[ -n "$LOG" ]] || LOG="logs/track_a/${RUN}_${TAG:-unknown}.log"
pkill -f "training.wandb.name=${RUN}" 2>/dev/null || true
sleep 2
if pgrep -af "training.wandb.name=${RUN}" | grep -q python; then
  echo "SKIP: already running"
  exit 0
fi
{
  echo ""
  echo "# === RESUME $(date -Is) post GPU restart ==="
  echo "# resume_from: ${RESUME_PT}"
  echo "# resume_global_step: ${GSTEP}"
  echo "# wandb_run_id: ${WID}"
} >>"$LOG"
nohup env PYTHONPATH=src PYTHONUNBUFFERED=1 WANDB_MODE=online \
  WANDB_RESUME=allow "WANDB_RUN_ID=${WID}" \
  .venv/bin/python -u -m smth2smth.pipelines.train \
  track=a "experiment=${EXP}" "seed=${SEED}" \
  "model.init_from=${SSL}" "model.head_queries=${Q}" model.tube_t=1 dataset.num_frames=4 \
  dataset.include_val_in_train=false dataset.official_val_holdout_ratio=0 \
  "training.checkpoint_path=checkpoints/track_a/videomaev2+ft/${RUN}.pt" \
  "training.resume_from=${RESUME_PT}" \
  "training.resume_global_step=${GSTEP}" \
  "training.wandb.project=smth2smth-frame-ablation" \
  ++training.wandb.group=ft-mae-scaling \
  "training.wandb.name=${RUN}" \
  ++training.wandb.config.head_type="perceiverQ${Q}" \
  ++training.wandb.config.pretrain_epochs="$(basename "$SSL" | sed -n 's/.*ep\([0-9]*\)\.pt/\1/p')" \
  ++training.wandb.config.num_queries="${Q}" \
  ++training.wandb.config.recipe=stab \
  >>"$LOG" 2>&1 &
echo "trainer_pid=$! log=$LOG"
REMOTE
}

# host:run:seed:ssl_ep:Q:wandb_id
Q8_ENTRIES=(
  anchois:perceiverQ8-mae100-s42:42:100:8:715f76zo
  labre:perceiverQ8-mae100-s43:43:100:8:vs4mcp2v
  truite:perceiverQ8-mae200-s42:42:200:8:culr1ccl
  thon:perceiverQ8-mae200-s43:43:200:8:4wte56ck
  lotte:perceiverQ8-mae300-s42:42:300:8:n6ibgn26
  sole:perceiverQ8-mae300-s43:43:300:8:jscsvpkp
  mulet:perceiverQ8-mae400-s42:42:400:8:x20b72jl
  murene:perceiverQ8-mae400-s43:43:400:8:zufhx731
)

do_resume() {
  bash "$REPO/scripts/resume_divspace_post_reboot_20260603.sh" resume
  bash "$REPO/scripts/launch_ft_mae_batch_20260602.sh" resume-meanpool
  for e in "${Q8_ENTRIES[@]}"; do
    IFS=: read -r H RUN SEED SSL_EP Q WID <<<"$e"
    resume_q8_stab "$H" "$RUN" "$SEED" "$SSL_EP" "$Q" "$WID" || echo "FAIL $H $RUN"
  done
}

do_health() {
  local ok=0 fail=0
  echo "=== Fleet health @ $(date -Is) ==="
  for h in ablette anchois anguille barbue brochet carrelet gardon labre lieu lotte mulet murene piranha raie requin rouget roussette sole thon truite; do
    echo "--- $h ---"
    if ssh -o BatchMode=yes "$h" "pgrep -af 'smth2smth.pipelines.train' | grep -v pgrep | head -1"; then
      ok=$((ok + 1))
    else
      echo "IDLE"
      fail=$((fail + 1))
    fi
  done
  echo "trainers: ok=$ok idle_or_fail=$fail"
}

case "$MODE" in
  resume) do_resume ;;
  chain-q16) bash "$REPO/scripts/resume_divspace_post_reboot_20260603.sh" chain-q16 ;;
  health) do_health ;;
  all)
    do_resume
    bash "$REPO/scripts/resume_divspace_post_reboot_20260603.sh" chain-q16
    echo "Waiting 90s for health..."
    sleep 90
    do_health || true
    ;;
  *) echo "Usage: $0 [all|resume|chain-q16|health]"; exit 1 ;;
esac
echo "Done ($MODE)."
