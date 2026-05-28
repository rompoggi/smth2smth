#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
# Home (~/.cache) is near the 30 GiB NFS quota; use /Data HF cache (see experiments_track_b_20052026.md).
export HF_HOME=/Data/thomas.turkieh/hf_cache
export TRANSFORMERS_CACHE=/Data/thomas.turkieh/hf_cache
export HUGGINGFACE_HUB_CACHE=/Data/thomas.turkieh/hf_cache/hub
cd /Data/thomas.turkieh/smth2smth
LOG=/Data/thomas.turkieh/smth2smth/logs/track_b_e1_ssv2ft_lora16f_cycle_20260523.log
{
  echo "[launcher] $(date -Is) waiting for free GPU (util<15%, mem<1500MiB)"
  while true; do
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' MiB')
    if [[ "${util}" -lt 15 && "${mem}" -lt 1500 ]]; then
      break
    fi
    echo "[launcher] $(date -Is) GPU busy util=${util}% mem=${mem}MiB; sleep 60s"
    sleep 60
  done
  echo "[launcher] $(date -Is) starting train (HF_HOME=${HF_HOME})"
} >>"${LOG}"
exec .venv/bin/python -u -m smth2smth.pipelines.train \
  track=b experiment=track_b_vjepa2_ssv2ft_lora16f >>"${LOG}" 2>&1
