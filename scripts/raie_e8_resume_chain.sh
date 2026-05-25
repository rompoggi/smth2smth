#!/usr/bin/env bash
# Resume E8 after interrupt: pretrain auto-resumes from espadon_encoder.state.pt,
# then champion FT when trunk checkpoint exists.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1 PYTHONPATH=src
PY="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing .venv — run: uv sync" >&2
  exit 1
fi

STATE="${REPO_ROOT}/checkpoints/track_a/ssl/espadon_encoder.state.pt"
if [[ ! -s "$STATE" ]]; then
  echo "Missing resume state: $STATE (need a prior pretrain run)" >&2
  exit 1
fi

echo "[chain] $(date -Is) Phase 1 RESUME VideoMAEv2 pretrain (auto from .state.pt)"
"$PY" -u -m smth2smth.pipelines.pretrain_videomae \
  experiment=track_a_ssl_pretrain_e8 track=a
test -s checkpoints/track_a/ssl/espadon_encoder.pt

echo "[chain] $(date -Is) Phase 2 champion FT ViT-B 60 ep"
"$PY" -u -m smth2smth.pipelines.train \
  experiment=track_a_ssl_finetune_e8 track=a
test -s checkpoints/track_a/ssl/espadon_ft.pt

echo "[chain] $(date -Is) Done."
