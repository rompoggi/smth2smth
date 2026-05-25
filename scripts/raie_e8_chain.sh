#!/usr/bin/env bash
# E8 (espadon): chained VideoMAEv2 ViT-B pretrain 250 ep + champion FT 60 ep.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1 PYTHONPATH=src
PY="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing .venv — run: uv sync" >&2
  exit 1
fi

BATCH_TAG="${BATCH_TAG:-$(date +%Y%m%d)}"
LOG="logs/raie_e8_chain_${BATCH_TAG}.log"
PID="logs/raie_e8_chain_${BATCH_TAG}.pid"

echo "[chain] $(date -Is) Phase 1 VideoMAEv2 pretrain ViT-B 250 ep (train+val+test)"
"$PY" -u -m smth2smth.pipelines.pretrain_videomae \
  experiment=track_a_ssl_pretrain_e8 track=a
test -s checkpoints/track_a/ssl/espadon_encoder.pt

echo "[chain] $(date -Is) Phase 2 champion FT ViT-B 60 ep"
"$PY" -u -m smth2smth.pipelines.train \
  experiment=track_a_ssl_finetune_e8 track=a
test -s checkpoints/track_a/ssl/espadon_ft.pt

echo "[chain] $(date -Is) Done."
