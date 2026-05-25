#!/usr/bin/env bash
# Resume E7 chain after crash/reboot: MAE pretrain from e7_encoder.pt, then FT.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1 PYTHONPATH=src
PY="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing .venv — run: uv sync" >&2
  exit 1
fi

ENCODER="${REPO_ROOT}/checkpoints/track_a/ssl/e7_encoder.pt"
test -s "$ENCODER" || { echo "Missing encoder checkpoint: $ENCODER" >&2; exit 1; }

echo "[chain] $(date -Is) Phase 1 MAE pretrain RESUME (E7, from ${ENCODER})"
"$PY" -u -m smth2smth.pipelines.pretrain_videomae \
  experiment=track_a_ssl_pretrain_e7 track=a \
  pretrain.resume_from="${ENCODER}"
test -s checkpoints/track_a/ssl/e7_encoder.pt

echo "[chain] $(date -Is) Phase 2 champion FT (60 ep)"
"$PY" -u -m smth2smth.pipelines.train \
  experiment=track_a_ssl_finetune_e7 track=a
test -s checkpoints/track_a/ssl/e7_ft.pt

echo "[chain] $(date -Is) Done."
