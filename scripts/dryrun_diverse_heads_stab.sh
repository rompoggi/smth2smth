#!/usr/bin/env bash
# Smoke-test all four Round-2 presets (1 epoch, 64 clips, W&B off).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
UV="${UV:-/users/eleves-a/2021/romain.poggi/.local/bin/uv}"
SSL="${SSL:-checkpoints/track_a/ssl/videomaev2_t4native_encoder_ep500.pt}"

if [[ ! -f "$SSL" ]]; then
  echo "ABORT: missing SSL encoder at $SSL"
  exit 1
fi

_dryrun() {
  local experiment="$1"
  shift
  local log="/tmp/dryrun_${experiment}.log"
  echo "======== dryrun: ${experiment} ========"
  PYTHONPATH=src WANDB_MODE=disabled "$UV" run python -u -m smth2smth.pipelines.train \
    experiment="${experiment}" \
    model.init_from="${SSL}" \
    dataset.max_samples=64 \
    training.epochs=1 \
    training.wandb.enabled=false \
    training.wandb.mode=disabled \
    training.early_stopping_enabled=false \
    test.num_segment=1 \
    test.num_crop=1 \
    "$@" 2>&1 | tee "$log" | tail -35
  if grep -qE 'Traceback|ConfigAttributeError|ConfigCompositionException' "$log"; then
    echo "FAILED: ${experiment} (see $log)"
    return 1
  fi
  if ! grep -q 'Done\. Best val top1' "$log"; then
    echo "FAILED: ${experiment} did not finish training (see $log)"
    return 1
  fi
  echo "OK: ${experiment}"
}

fail=0
_dryrun track_a_diverse_arch2_perceiver_stab \
  training.checkpoint_path=checkpoints/track_a/videomaev2+ft/_dryrun_arch2-stab.pt \
  || fail=1

_dryrun track_a_diverse_arch3_divided_st_stab \
  training.checkpoint_path=checkpoints/track_a/videomaev2+ft/_dryrun_arch3-k6-stab.pt \
  || fail=1

_dryrun track_a_diverse_arch3_divided_st_k3_stab \
  training.checkpoint_path=checkpoints/track_a/videomaev2+ft/_dryrun_arch3-k3-stab.pt \
  || fail=1

_dryrun track_a_diverse_arch2_perceiver_stab_s42 \
  seed=123 \
  training.checkpoint_path=checkpoints/track_a/videomaev2+ft/_dryrun_arch2-stab-s42.pt \
  || fail=1

if [[ "$fail" -ne 0 ]]; then
  echo "One or more dryruns failed."
  exit 1
fi
echo "All four stabilized dryruns passed (control omitted — already have ep500 mean-pool)."
