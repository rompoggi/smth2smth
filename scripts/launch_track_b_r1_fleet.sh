#!/usr/bin/env bash
# After configs are on origin/main: verify + launch Track B r1 grid on 12 fish hosts.
# Run from gymnote:  bash scripts/launch_track_b_r1_fleet.sh [verify|launch|all]
set -euo pipefail
REPO=/Data/romain.poggi/smth2smth
cd "$REPO"

HOSTS=(anguille barbeau barbue baudroie gardon lieu murene piranha raie roussette saumon truite)
RUNS=(r1_b01_hf_lr5e3 r1_b02_hf_lr3e3 r1_b03_hf_lr1e3 r1_b04_hf_lr3e4 r1_b05_hf_lr1e4 r1_b06_hf_wd001 r1_b07_mq16_lr1e3 r1_b08_mq16_lr3e3 r1_b09_mq8_lr1e3 r1_b10_mq32_lr1e3 r1_b11_4block_lr3e4 r1_b12_hf_dora_mlp)
TAG=$(date +%Y%m%d)
MODE="${1:-all}"

prep_host() {
  local H="$1"
  echo "=== PREP $H ==="
  ssh -o BatchMode=yes "$H" bash -s <<'PREP'
set -euo pipefail
DATA=/Data/romain.poggi
REPO=/Data/romain.poggi/smth2smth
mkdir -p "$DATA"
chmod 700 "$DATA"
cd "$REPO"
git pull --ff-only 2>/dev/null || git pull
if [[ ! -x .venv/bin/python ]]; then
  uv sync
fi
PREP
  ssh -o BatchMode=yes "$H" "bash $REPO/scripts/verify_track_b_r1_host.sh"
}

launch_one() {
  local H="$1" R="$2"
  local E="track_b_${R}"
  local LOG="logs/track_b/${R}_${TAG}.log"
  echo "=== LAUNCH $H -> $R ==="
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$R" "$E" "$LOG" <<'LAUNCH'
set -euo pipefail
REPO="$1"
R="$2"
E="$3"
LOG="$4"
cd "$REPO"
mkdir -p logs/track_b checkpoints/track_b
if pgrep -af "experiment=${E}" >/dev/null 2>&1; then
  echo "SKIP: already running ${E} on $(hostname)"
  exit 0
fi
{
  echo "# run: ${R}"
  echo "# started: $(date -Is)"
  echo "# track: b"
  echo "# experiment_doc: track_b_round1.md"
  echo "# hydra: experiment=${E}"
  echo "# host: $(hostname)"
} >"$LOG"
nohup env PYTHONPATH=src PYTHONUNBUFFERED=1 .venv/bin/python -u -m smth2smth.pipelines.train \
  track=b "experiment=${E}" \
  "training.checkpoint_path=checkpoints/track_b/${R}.pt" \
  >>"$LOG" 2>&1 &
echo "pid=$! log=$LOG"
LAUNCH
}

case "$MODE" in
  verify)
    for H in "${HOSTS[@]}"; do prep_host "$H" || echo "FAIL $H"; done
    ;;
  launch)
    for i in "${!RUNS[@]}"; do launch_one "${HOSTS[$i]}" "${RUNS[$i]}"; done
    ;;
  all)
    for H in "${HOSTS[@]}"; do
      prep_host "$H" || { echo "Aborting: $H not ready"; exit 1; }
    done
    for i in "${!RUNS[@]}"; do launch_one "${HOSTS[$i]}" "${RUNS[$i]}"; done
    ;;
  *)
    echo "Usage: $0 [verify|launch|all]"; exit 1
    ;;
esac
echo "Done ($MODE)."
