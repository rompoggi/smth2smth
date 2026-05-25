#!/usr/bin/env bash
# Val90 holdout classifier-only FT from Run 9 best checkpoint (T=16, LoRA r=16).
#
# Default: print the launch command only (no training). To start under nohup:
#   START=1 ./scripts/run_track_b_vjepa2_hfclf_16f_lora_r16_val90_holdout_clf.sh
#
# Optional overrides:
#   BATCH_TAG=20260524_val90clf   # log/pid filename stamp
#   WANDB_ENTITY=my-team          # passed as training.wandb_entity=...
#   EXTRA_OVERRIDES='training.epochs=12'
#   RESUME=1                      # continue from .last.pt (epoch/optim/scheduler)
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
EXPT="track_b_vjepa2_hfclf_16f_lora_r16_val90_holdout_clf"
CKPT_SRC="checkpoints/track_b/vitl_fpc16ssv2_16f_lora_r16.pt"
CKPT_LAST="checkpoints/track_b/vitl_fpc16ssv2_16f_lora_r16_val90_holdout_clf.last.pt"
CKPT_FT="checkpoints/track_b/vitl_fpc16ssv2_16f_lora_r16_val90_holdout_clf.pt"
TAG="${BATCH_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG="logs/track_b_vjepa2_hfclf_16f_lora_r16_val90_holdout_clf_${TAG}.log"
PID_FILE="logs/track_b_vjepa2_hfclf_16f_lora_r16_val90_holdout_clf_${TAG}.pid"
HYDRA_DIR="outputs/track_b_vjepa2_hfclf_16f_lora_r16_val90_holdout_clf_${TAG}"

WANDB_EXTRA=()
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  WANDB_EXTRA+=("training.wandb_entity=${WANDB_ENTITY}")
fi

CMD=(
  env PYTHONUNBUFFERED=1 PYTHONPATH=src
  "${PY}" -u -m smth2smth.pipelines.train
  track=b "experiment=${EXPT}"
  "training.checkpoint_path=${CKPT_FT}"
  "hydra.run.dir=${HYDRA_DIR}"
)
if [[ "${RESUME:-0}" == "1" ]]; then
  if [[ ! -f "${CKPT_LAST}" ]]; then
    echo "ERROR: RESUME=1 but missing ${CKPT_LAST}" >&2
    exit 1
  fi
  # Quote path: Hydra treats unquoted ``.last`` in filenames as nested keys.
  CMD+=(
    "training.resume_from='${CKPT_LAST}'"
    training.resume_reset_epoch=false
    training.resume_apply_cfg_lr=false
  )
fi
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA=( ${EXTRA_OVERRIDES} )
  CMD+=( "${EXTRA[@]}" )
fi
CMD+=( "${WANDB_EXTRA[@]}" )

echo "=== Track B Run 9 val90 holdout clf FT (prepared, not started unless START=1) ==="
echo "Experiment: ${EXPT}"
echo "Resume from: ${CKPT_SRC}"
echo "Save to:     ${CKPT_FT}"
echo "Log:         ${LOG}"
echo "PID file:    ${PID_FILE}"
echo "Hydra dir:   ${HYDRA_DIR}"
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
  echo "Detached (survives logout; script writes ${LOG} — do not redirect stdout):"
  echo "  START=1 BATCH_TAG=${TAG} nohup $0 < /dev/null > /dev/null 2>&1 &"
  echo "  echo \$! > ${PID_FILE}"
  exit 0
fi

if [[ "${RESUME:-0}" == "1" ]]; then
  if [[ ! -f "${CKPT_LAST}" ]]; then
    echo "ERROR: RESUME=1 but missing ${CKPT_LAST}" >&2
    exit 1
  fi
elif [[ ! -f "${CKPT_SRC}" ]]; then
  echo "ERROR: missing source checkpoint ${CKPT_SRC}" >&2
  exit 1
fi

mkdir -p logs checkpoints
echo $$ > "${PID_FILE}"
if [[ -t 1 ]]; then
  echo "[$(date -Is)] starting val90 holdout clf FT -> ${CKPT_FT}" | tee "${LOG}"
  "${CMD[@]}" 2>&1 | tee -a "${LOG}"
else
  {
    if [[ "${RESUME:-0}" == "1" ]]; then
      echo "[$(date -Is)] resuming val90 holdout clf FT from ${CKPT_LAST} -> ${CKPT_FT}"
    else
      echo "[$(date -Is)] starting val90 holdout clf FT -> ${CKPT_FT}"
    fi
    "${CMD[@]}"
  } >> "${LOG}" 2>&1
fi
