#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1 PYTHONPATH=src
PY="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing .venv — run: uv sync" >&2
  exit 1
fi

echo "[chain] $(date -Is) Phase 1 MAE pretrain (E7, mask schedule 0.90→0.75, 150 ep)"
"$PY" -u -m smth2smth.pipelines.pretrain_videomae \
  experiment=track_a_ssl_pretrain_e7 track=a
test -s checkpoints/track_a/ssl/e7_encoder.pt

echo "[chain] $(date -Is) Phase 2 champion FT (60 ep)"
"$PY" -u -m smth2smth.pipelines.train \
  experiment=track_a_ssl_finetune_e7 track=a
test -s checkpoints/track_a/ssl/e7_ft.pt

echo "[chain] $(date -Is) Done."
