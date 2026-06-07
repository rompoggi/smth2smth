#!/usr/bin/env bash
# Local (gymnote) smoke tests before git push + fleet pull on other hosts.
# Runs mean-pool + Perceiver Q16 canaries (256 samples, 1 epoch, seed 43).
set -euo pipefail

REPO="${REPO:-/Data/romain.poggi/smth2smth}"
cd "$REPO"
TAG="${TAG:-$(date +%Y%m%d)}"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-online}"

PY="${PY:-.venv/bin/python}"
mkdir -p logs/track_a checkpoints/track_a/videomaev2+ft

run_canary() {
  local RUN="$1" EXP="$2" SSL="$3" HEAD="$4" EPOCHS="$5"
  local LOG="logs/track_a/canary-${RUN}_${TAG}.log"
  if pgrep -af "experiment=${EXP}.*seed=43.*${RUN}" >/dev/null 2>&1; then
    echo "SKIP: already running ${RUN}"
    return 0
  fi
  : >"$LOG"
  {
    echo "# run: canary-${RUN}"
    echo "# started: $(date -Is)"
    echo "# track: a"
    echo "# experiment_doc: experiments/experimental_cleanup.md"
    echo "# hydra: experiment=${EXP} seed=43 T=4"
    echo "# host: $(hostname)"
    echo "# ssl: ${SSL}"
  } >>"$LOG"
  nohup "${PY}" -u -m smth2smth.pipelines.train \
    track=a "experiment=${EXP}" seed=43 \
    "model.init_from=${SSL}" model.tube_t=1 dataset.num_frames=4 \
    dataset.include_val_in_train=false dataset.official_val_holdout_ratio=0 \
    "training.checkpoint_path=checkpoints/track_a/videomaev2+ft/canary-${RUN}.pt" \
    "training.wandb.project=smth2smth-frame-ablation" \
    ++training.wandb.group=ft-mae-scaling \
    "training.wandb.name=${RUN}" \
    ++training.wandb.config.head_type="${HEAD}" \
    ++training.wandb.config.pretrain_epochs="${EPOCHS}" \
    dataset.max_samples=256 training.epochs=1 \
    >>"$LOG" 2>&1 &
  echo "started ${RUN} pid=$! log=${LOG}"
}

health_log() {
  local LOG="$1"
  tail -15 "$LOG"
  grep -q Traceback "$LOG" && return 1
  grep -q '\[init_from\] loaded' "$LOG" || return 1
  grep -q '\[wandb\] run started:' "$LOG" || return 1
  grep -qE '\[.*\] step [0-9]+/' "$LOG" || return 1
  grep -q 'Done\. Best' "$LOG" || return 1
  echo "OK ${LOG}"
}

SSL100="${REPO}/checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep100.pt"
SSL50="${REPO}/checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep50.pt"

MODE="${1:-all}"
case "$MODE" in
  meanpool)
    run_canary meanpool-mae100-s43 track_a_videomae_official_ssv2_ft "$SSL100" meanpool 100
    ;;
  perceiver)
    run_canary perceiverQ16-mae100-s43 track_a_diverse_arch2_perceiver "$SSL100" perceiverQ16 100
    ;;
  launch)
    run_canary meanpool-mae050-s43 track_a_videomae_official_ssv2_ft "$SSL50" meanpool 50
    run_canary meanpool-mae100-s43 track_a_videomae_official_ssv2_ft "$SSL100" meanpool 100
    run_canary perceiverQ16-mae100-s43 track_a_diverse_arch2_perceiver "$SSL100" perceiverQ16 100
    ;;
  health)
    health_log "logs/track_a/canary-meanpool-mae100-s43_${TAG}.log" || exit 1
    health_log "logs/track_a/canary-perceiverQ16-mae100-s43_${TAG}.log" || exit 1
    ;;
  all)
    run_canary meanpool-mae100-s43 track_a_videomae_official_ssv2_ft "$SSL100" meanpool 100
    echo "Waiting for meanpool canary..."
    for _ in $(seq 1 90); do
      grep -q 'Done\. Best' "logs/track_a/canary-meanpool-mae100-s43_${TAG}.log" 2>/dev/null && break
      sleep 2
    done
    health_log "logs/track_a/canary-meanpool-mae100-s43_${TAG}.log"
    run_canary perceiverQ16-mae100-s43 track_a_diverse_arch2_perceiver "$SSL100" perceiverQ16 100
    echo "Waiting for perceiver canary..."
    for _ in $(seq 1 120); do
      grep -q 'Done\. Best' "logs/track_a/canary-perceiverQ16-mae100-s43_${TAG}.log" 2>/dev/null && break
      sleep 2
    done
    health_log "logs/track_a/canary-perceiverQ16-mae100-s43_${TAG}.log"
    echo "Both local canaries passed."
    ;;
  *)
    echo "Usage: $0 [meanpool|perceiver|launch|health|all]"; exit 1
    ;;
esac
