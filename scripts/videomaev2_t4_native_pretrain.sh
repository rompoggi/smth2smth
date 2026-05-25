#!/usr/bin/env bash
# VideoMAE v2 ViT-B SSL pretrain — native T=4, 300 ep, train+val+test.
# Spec: videomaev2_t4_native_300e_pretrain.md
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
  echo "missing .venv — run: uv sync" >&2
  exit 1
fi

BATCH_TAG="${BATCH_TAG:-$(date +%Y%m%d)}"
LOG_DIR="${REPO_ROOT}/logs/pre_train"
mkdir -p "$LOG_DIR" "${REPO_ROOT}/checkpoints/track_a/ssl"
LOG="${LOG_DIR}/videomaev2_t4native_pretrain_${BATCH_TAG}.log"

# Single sink for all output (do not also redirect this script via nohup >> "$LOG").
{
  echo "[videomaev2_t4native] $(date -Is) pretrain ViT-B 500 ep T=4 native bs=96 (train+val+test)"
  "$PY" -u -m smth2smth.pipelines.pretrain_videomae \
    experiment=track_a_ssl_pretrain_e9_t4_native track=a
  test -s checkpoints/track_a/ssl/videomaev2_t4native_encoder.pt
  echo "[videomaev2_t4native] $(date -Is) Done."
} >>"$LOG" 2>&1
