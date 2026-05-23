#!/usr/bin/env bash
# Launch Track-A VideoMAE official SSv2 fine-tuning (recipe preset + W&B).
#
# The **recipe** is fixed in configs/experiment/track_a_videomae_official_ssv2_ft.yaml.
# Swap SSL encoder / run label via environment variables (or pass Hydra overrides yourself).
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  Variable          │  Role                                                  │
# ├────────────────────┼────────────────────────────────────────────────────────┤
# │  SSL_ENCODER       │  Input: SSL trunk .pt (must contain trunk_state_dict)  │
# │  RUN_NAME          │  W&B run name + FT checkpoint basename (*.pt)          │
# │  WANDB_PROJECT     │  W&B project (default: smth2smth-frame-ablation)     │
# │  NUM_FRAMES        │  Clip length — must match encoder pretrain (default 16)│
# │  TUBE_T            │  Must match encoder (default 1 for espadon/t16)        │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# Examples:
#
#   # Current espadon run (defaults):
#   ./scripts/launch_videomae_official_ssv2_ft.sh
#
#   # Same recipe, another encoder:
#   SSL_ENCODER=checkpoints/track_a/ssl/e_r1_encoder.pt \
#   RUN_NAME=e_r1-official-ft-f4 \
#   NUM_FRAMES=4 TUBE_T=1 \
#   ./scripts/launch_videomae_official_ssv2_ft.sh
#
#   # Foreground (no nohup):
#   NOHUP=0 ./scripts/launch_videomae_official_ssv2_ft.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# --- Run identity (override these) -------------------------------------------
SSL_ENCODER="${SSL_ENCODER:-${REPO_ROOT}/checkpoints/track_a/ssl/espadon_t16_encoder_ep50.pt}"
RUN_NAME="${RUN_NAME:-mae50-ft-f16}"
NUM_FRAMES="${NUM_FRAMES:-16}"
TUBE_T="${TUBE_T:-1}"

# --- W&B (.env may set WANDB_API_KEY; project overridable here) --------------
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-smth2smth-frame-ablation}"

# --- Outputs -----------------------------------------------------------------
FT_CHECKPOINT="${FT_CHECKPOINT:-${REPO_ROOT}/checkpoints/track_a/videomaev2+ft/${RUN_NAME}.pt}"
TAG="${TAG:-$(date +%Y%m%d)}"
LOG="${LOG:-logs/${RUN_NAME}_${TAG}.log}"
PIDFILE="${PIDFILE:-logs/${RUN_NAME}_${TAG}.pid}"
NOHUP="${NOHUP:-1}"

mkdir -p logs checkpoints/track_a/videomaev2+ft

if [[ ! -f "${SSL_ENCODER}" ]]; then
  echo "error: SSL_ENCODER not found: ${SSL_ENCODER}" >&2
  exit 1
fi

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export PATH="${HOME}/.local/bin:${PATH}"

UV_BIN="${UV_BIN:-$(command -v uv)}"
if [[ -z "${UV_BIN}" || ! -x "${UV_BIN}" ]]; then
  echo "error: uv not found; set UV_BIN or add ~/.local/bin to PATH." >&2
  exit 1
fi

EXPERIMENT="${EXPERIMENT:-track_a_videomae_official_ssv2_ft}"

HYDRA_ARGS=(
  track=a
  "experiment=${EXPERIMENT}"
  "model.init_from=${SSL_ENCODER}"
  "model.tube_t=${TUBE_T}"
  "dataset.num_frames=${NUM_FRAMES}"
  "training.checkpoint_path=${FT_CHECKPOINT}"
  "training.wandb.name=${RUN_NAME}"
  "training.wandb.project=${WANDB_PROJECT}"
)
# Optional extra Hydra overrides, e.g. dataset.max_samples=64 for dry-run.
if [[ $# -gt 0 ]]; then
  HYDRA_ARGS+=("$@")
fi

echo "=== VideoMAE official SSv2 fine-tune ==="
echo "experiment     : ${EXPERIMENT}"
echo "SSL encoder    : ${SSL_ENCODER}"
echo "FT checkpoint  : ${FT_CHECKPOINT}"
echo "frames / tube_t: ${NUM_FRAMES} / ${TUBE_T}"
echo "W&B            : ${WANDB_PROJECT} / ${RUN_NAME} (mode=${WANDB_MODE})"
echo "log / pid      : ${LOG} / ${PIDFILE}"
echo "Hydra          : ${HYDRA_ARGS[*]}"
echo ""

run_train() {
  "${UV_BIN}" run python -m smth2smth.pipelines.train "${HYDRA_ARGS[@]}"
}

if [[ "${NOHUP}" == "1" ]]; then
  nohup run_train >"${LOG}" 2>&1 &
  echo $! >"${PIDFILE}"
  echo "Started PID $(cat "${PIDFILE}") — tail -f ${LOG}"
else
  run_train 2>&1 | tee "${LOG}"
fi
