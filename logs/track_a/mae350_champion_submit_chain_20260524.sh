#!/usr/bin/env bash
# Wait for honest-FT champion submit, then run val90-holdout champion submit.
set -euo pipefail
REPO=/Data/romain.poggi/smth2smth
cd "$REPO"
export PATH="/users/eleves-a/2021/romain.poggi/.local/bin:${PATH}"
export PYTHONPATH=src PYTHONUNBUFFERED=1
UV=/users/eleves-a/2021/romain.poggi/.local/bin/uv
PIDFILE=logs/mae350-ft-f4_submit_champion_20260524.pid
LOG=logs/mae350_champion_submit_chain_20260524.log

if [[ -f "$PIDFILE" ]]; then
  echo "[chain] waiting for PID $(cat "$PIDFILE") ..." | tee -a "$LOG"
  while kill -0 "$(cat "$PIDFILE")" 2>/dev/null; do sleep 60; done
fi
echo "[chain] starting val90 holdout champion submit $(date -Is)" | tee -a "$LOG"
exec "$UV" run python -u -m smth2smth.pipelines.submit \
  track=a experiment=track_a_videomae_submit_champion dataset.num_frames=4 \
  training.checkpoint_path=checkpoints/track_a/videomaev2+ft/mae350-ft-f4_val90_holdout.pt \
  dataset.submission_output=submissions/track_a_mae350_ft_f4_val90_holdout_champion_tta_20260524.csv \
  2>&1 | tee -a logs/mae350-ft-f4-val90_submit_champion_20260524.log
