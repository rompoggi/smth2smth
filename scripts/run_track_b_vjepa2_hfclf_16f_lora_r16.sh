#!/usr/bin/env bash
# Track B Run 9: V-JEPA2 ViT-L SSv2 checkpoint, T=16 (4-frame duplicate), LoRA r=16.
#
# Default: print the launch command only (no training). To start under nohup:
#   START=1 ./scripts/run_track_b_vjepa2_hfclf_16f_lora_r16.sh
#
# Optional overrides:
#   BATCH_TAG=20260522_1200   # log/pid filename stamp (default: date +%Y%m%d_%H%M%S)
#   WANDB_ENTITY=my-team       # passed as training.wandb_entity=...
#   EXTRA_OVERRIDES='training.epochs=20'  # extra Hydra CLI fragments
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
EXPT="track_b_vjepa2_hfclf_16f_lora_r16"
CKPT="checkpoints/track_b/vitl_fpc16ssv2_16f_lora_r16.pt"
TAG="${BATCH_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG="logs/track_b_vjepa2_hfclf_16f_lora_r16_${TAG}.log"
PID_FILE="logs/track_b_vjepa2_hfclf_16f_lora_r16_${TAG}.pid"
HYDRA_DIR="outputs/track_b_vjepa2_hfclf_16f_lora_r16_${TAG}"

WANDB_EXTRA=()
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  WANDB_EXTRA+=("training.wandb_entity=${WANDB_ENTITY}")
fi

CMD=(
  env PYTHONUNBUFFERED=1 PYTHONPATH=src
  "${PY}" -u -m smth2smth.pipelines.train
  track=b "experiment=${EXPT}"
  "training.checkpoint_path=${CKPT}"
  "hydra.run.dir=${HYDRA_DIR}"
)
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA=( ${EXTRA_OVERRIDES} )
  CMD+=( "${EXTRA[@]}" )
fi
CMD+=( "${WANDB_EXTRA[@]}" )

echo "=== Track B Run 9 (prepared, not started unless START=1) ==="
echo "Experiment: ${EXPT}"
echo "Checkpoint: ${CKPT}"
echo "Log:        ${LOG}"
echo "PID file:   ${PID_FILE}"
echo "Hydra dir:  ${HYDRA_DIR}"
echo "W&B:        project=smth2smth-track-b name=vitl_fpc16ssv2_16f_lora_r16"
echo ""
echo "Launch command:"
printf '  %q ' "${CMD[@]}"
echo ">"
echo "  ${LOG}"
echo ""

if [[ "${START:-0}" != "1" ]]; then
  echo "Dry-run only. To start training:"
  echo "  START=1 $0"
  echo ""
  echo "Detached (survives logout):"
  echo "  START=1 nohup $0 >> ${LOG} 2>&1 < /dev/null &"
  echo "  echo \$! > ${PID_FILE}"
  exit 0
fi

mkdir -p logs checkpoints
echo "[$(date -Is)] starting Run 9 -> ${LOG}" | tee "${LOG}"
echo $$ > "${PID_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${LOG}"
