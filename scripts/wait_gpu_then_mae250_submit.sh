#!/usr/bin/env bash
# Wait until the GPU has enough free memory, then run mae250-ft-f4 champion-TTA submit.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MIN_FREE_MIB="${MIN_FREE_MIB:-18000}"
POLL_SEC="${POLL_SEC:-60}"
LOG="${LOG:-logs/mae250-ft-f4_submit_tta3flip_20260524.log}"
PIDFILE="${PIDFILE:-logs/mae250-ft-f4_submit_tta3flip_20260524.pid}"
SUBMIT_PIDFILE="${SUBMIT_PIDFILE:-logs/mae250-ft-f4_submit_tta3flip_20260524.waiter.pid}"

echo $$ >"${SUBMIT_PIDFILE}"
echo "[$(date -Is)] waiter PID $$ — need >= ${MIN_FREE_MIB} MiB free GPU memory" | tee -a "${LOG}"

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
PY="${REPO_ROOT}/.venv/bin/python"

while true; do
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  if [[ -n "${free_mib}" && "${free_mib}" -ge "${MIN_FREE_MIB}" ]]; then
    echo "[$(date -Is)] GPU free ${free_mib} MiB >= ${MIN_FREE_MIB} — starting submit" | tee -a "${LOG}"
    break
  fi
  echo "[$(date -Is)] GPU free ${free_mib:-?} MiB — waiting ${POLL_SEC}s" | tee -a "${LOG}"
  sleep "${POLL_SEC}"
done

"${PY}" -u -m smth2smth.pipelines.submit \
  track=a \
  experiment=track_a_videomae_official_ssv2_ft \
  dataset.num_frames=4 \
  training.checkpoint_path=checkpoints/track_a/videomaev2+ft/mae250-ft-f4.pt \
  training.tta=true \
  training.tta_flip=true \
  'training.tta_scales=[0.857,1.0,1.143]' \
  test.num_segment=1 \
  test.num_crop=1 \
  test.flip_tta=false \
  training.batch_size=4 \
  dataset.submission_output=submissions/track_a_mae250-ft-f4_tta3flip_20260524.csv \
  2>&1 | tee -a "${LOG}"

echo "[$(date -Is)] submit finished (exit ${PIPESTATUS[0]})" | tee -a "${LOG}"
