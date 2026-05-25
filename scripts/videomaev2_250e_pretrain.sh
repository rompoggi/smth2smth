#!/usr/bin/env bash
# VideoMAEv2 ViT-B SSL pretrain 250 ep @ T=16 — see experiments/videomaev2_250e_pretrain.md
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1 PYTHONPATH=src

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/.env"
  set +a
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "Missing WANDB_API_KEY — add it to ${REPO_ROOT}/.env" >&2
  exit 1
fi

PY="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing .venv — run: uv sync" >&2
  exit 1
fi

BATCH_TAG="${BATCH_TAG:-$(date +%Y%m%d)}"
LOG_DIR="${REPO_ROOT}/logs/pre_train"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/videomaev2_250e_pretrain_${BATCH_TAG}.log"
PID="${LOG_DIR}/videomaev2_250e_pretrain_${BATCH_TAG}.pid"

echo "[videomaev2] $(date -Is) pretrain ViT-B 250 ep T=16 (train+val+test, interp 4->16)"
"$PY" -u -m smth2smth.pipelines.pretrain_videomae \
  experiment=track_a_ssl_pretrain_e8_t16 track=a
test -s checkpoints/track_a/ssl/espadon_t16_encoder.pt

echo "[videomaev2] $(date -Is) Done."
