#!/usr/bin/env bash
# Preflight for one Track B r1 host. Exit 0 = ready to train.
set -euo pipefail
DATA=/Data/romain.poggi
REPO=/Data/romain.poggi/smth2smth
cd "$REPO"

PERM=$(stat -c '%a' "$DATA")
[[ "$PERM" == "700" ]] || { echo "FAIL: $DATA mode=$PERM (want 700)"; exit 1; }

git rev-parse --git-dir >/dev/null
test -f configs/experiment/track_b_r1_b01_hf_lr5e3.yaml || {
  echo "FAIL: track_b_r1 experiment YAMLs missing (git pull after push?)"
  exit 1
}

test -f data/holdout_clean.json || {
  echo "FAIL: data/holdout_clean.json missing (run scripts/build_holdout_clean.py after pull)"
  exit 1
}

grep -qE '^WANDB_API_KEY=.+$' .env || { echo "FAIL: WANDB_API_KEY missing in .env"; exit 1; }

if [[ ! -x .venv/bin/python ]]; then
  echo "[verify] running uv sync..."
  uv sync
fi
test -x .venv/bin/python || { echo "FAIL: .venv/bin/python missing"; exit 1; }

N=$(find data/train -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
[[ "$N" -ge 32 ]] || { echo "FAIL: data/train has $N class dirs (want >=32)"; exit 1; }

PYTHONPATH=src .venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"

echo "OK $(hostname -s): perm=$PERM wandb=yes train_classes=$N cuda=yes r1_configs=yes"
