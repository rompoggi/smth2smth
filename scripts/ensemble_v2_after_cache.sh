#!/usr/bin/env bash
# Wait for v2 holdout caches, optimize; after full cache (incl. test), submit best.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PIDFILE=logs/ensemble_v2_cache_20260519.pid
LOG=logs/ensemble_v2_optimize_20260519.log
UV=/users/eleves-a/2021/romain.poggi/.local/bin/uv
CACHE=outputs/ensemble/videomaev2_3seed_v2

holdout_ready() {
  for s in 42 43 44; do
    [[ -f "$CACHE/logits_s${s}_official_2x3_logits.npy" ]] || return 1
    [[ -f "$CACHE/probs_s${s}_official_2x3_probs.npy" ]] || return 1
  done
  [[ -f "$CACHE/labels_holdout.npy" ]]
}

echo "[after_cache] waiting for holdout official_2x3 caches ..." | tee -a "$LOG"
while ! holdout_ready; do
  sleep 60
done

{
  echo "[after_cache] holdout caches ready at $(date -Is)"
  env PYTHONUNBUFFERED=1 PYTHONPATH=src "$UV" run python -u scripts/ensemble_track_a_videomae.py optimize \
    --cache-dir "$CACHE"
} >>"$LOG" 2>&1

while kill -0 "$(cat "$PIDFILE")" 2>/dev/null; do
  sleep 120
done

{
  echo "[after_cache] full cache done at $(date -Is)"
  env PYTHONUNBUFFERED=1 PYTHONPATH=src "$UV" run python -u scripts/ensemble_track_a_videomae.py submit \
    --cache-dir "$CACHE" --exp best \
    --output submissions/track_a_ensemble_v2_best.csv
  echo "[after_cache] submit done at $(date -Is)"
} >>"$LOG" 2>&1
