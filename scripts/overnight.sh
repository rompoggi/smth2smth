#!/usr/bin/env bash
set -u  # do NOT use -e: a single failed sweep job must not abort the whole night

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT/src"

EPOCHS="${EPOCHS:-20}"
LRS_A="${LRS_A:-1e-3,5e-4,1e-4}"
LRS_B="${LRS_B:-5e-4,1e-4,5e-5}"

mkdir -p checkpoints/runs submissions logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Repo: $REPO_ROOT"
log "EPOCHS=$EPOCHS  LRS_A=$LRS_A  LRS_B=$LRS_B"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
fi

log "=== Track A x {augment=none, augment=strong} x {lr in $LRS_A} ==="
uv run python scripts/run_track_a.py -m \
    augment=none,strong \
    training.lr="$LRS_A" \
    training.epochs="$EPOCHS" \
    'training.checkpoint_path=${hydra:runtime.cwd}/checkpoints/runs/track_a_aug-${augment.name}_lr${training.lr}.pt' \
    || log "Track A multirun returned non-zero. Continuing with Track B."

log "=== Track B x {augment=none, augment=strong} x {lr in $LRS_B} ==="
uv run python scripts/run_track_b.py -m \
    augment=none,strong \
    training.lr="$LRS_B" \
    training.epochs="$EPOCHS" \
    'training.checkpoint_path=${hydra:runtime.cwd}/checkpoints/runs/track_b_aug-${augment.name}_lr${training.lr}.pt' \
    || log "Track B multirun returned non-zero."

log "Done. Checkpoints under: $REPO_ROOT/checkpoints/runs/"
ls -lh checkpoints/runs/ 2>/dev/null || true
