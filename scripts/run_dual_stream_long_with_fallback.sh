#!/usr/bin/env bash
# Track A: dual-stream RGB + frame-difference TSM long run with automatic
# slower-LR fallback. Sequential workflow:
#   1. Train with LR1 (default 1e-4) and motion_lr_ratio=0.3 on the
#      ``track_a_dual_stream_long_class_boost`` preset (cosine, warmup,
#      early stopping patience=3 on val top-1, class boosting).
#   2. If the first run wrote ``Early stopping triggered`` to its log,
#      launch a second run with LR2 (default 0.6 x LR1) to a fresh
#      checkpoint / log file.
#
# All paths are repo-relative. Logs go to ``logs/`` and checkpoints to
# ``checkpoints/track_a/``. Each run also drops a ``.pid`` file next to
# its log so the user can ``kill -0 $(cat ...)`` to monitor it.
#
# Usage (foreground, blocks until both runs are done):
#   bash scripts/run_dual_stream_long_with_fallback.sh
#
# Usage (background; survives logout):
#   nohup bash scripts/run_dual_stream_long_with_fallback.sh \
#       > logs/dual_stream_long_with_fallback_$(date +%Y%m%d_%H%M%S).out 2>&1 &

set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LR1="${LR1:-0.0001}"
LR2="${LR2:-0.00006}"
MOTION_LR_RATIO="${MOTION_LR_RATIO:-0.3}"
EXPERIMENT="${EXPERIMENT:-track_a_dual_stream_long_class_boost}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

mkdir -p logs checkpoints/track_a

stamp="$(date +%Y%m%d_%H%M%S)"
log1="logs/track_a_dual_stream_long_lr1_${stamp}.log"
pid1="logs/track_a_dual_stream_long_lr1_${stamp}.pid"
ckpt1="checkpoints/track_a/dual_stream_long_class_boost_lr1_${stamp}.pt"
hydra_dir1="outputs/track_a_dual_stream_long_lr1_${stamp}"

echo "[wrapper] $(date -Is) starting Run 1: lr=${LR1}, motion_lr_ratio=${MOTION_LR_RATIO}"
echo "[wrapper] log:  ${log1}"
echo "[wrapper] ckpt: ${ckpt1}"

PYTHONPATH=src PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u -m smth2smth.pipelines.train \
    track=a \
    experiment="${EXPERIMENT}" \
    training.lr="${LR1}" \
    training.motion_lr_ratio="${MOTION_LR_RATIO}" \
    training.checkpoint_path="${REPO_ROOT}/${ckpt1}" \
    hydra.run.dir="${hydra_dir1}" \
    > "${log1}" 2>&1 < /dev/null &
child_pid1=$!
echo "${child_pid1}" > "${pid1}"
echo "[wrapper] Run 1 launched as PID ${child_pid1} (pidfile ${pid1})"

# Block until Run 1 finishes (the wrapper itself is the thing that should
# survive logout via nohup; Python is a child of this script).
wait "${child_pid1}"
exit_code1=$?
echo "[wrapper] $(date -Is) Run 1 finished with exit code ${exit_code1}"

if [[ "${exit_code1}" -ne 0 ]]; then
    echo "[wrapper] Run 1 exited non-zero; SKIPPING fallback (fix the failure first)."
    exit "${exit_code1}"
fi

if grep -q "Early stopping triggered" "${log1}"; then
    echo "[wrapper] Run 1 hit early stopping. Launching Run 2 with slower LR=${LR2}."
else
    echo "[wrapper] Run 1 ran to the configured epoch budget (no early stopping); skipping Run 2."
    exit 0
fi

stamp2="$(date +%Y%m%d_%H%M%S)"
log2="logs/track_a_dual_stream_long_lr2_${stamp2}.log"
pid2="logs/track_a_dual_stream_long_lr2_${stamp2}.pid"
ckpt2="checkpoints/track_a/dual_stream_long_class_boost_lr2_${stamp2}.pt"
hydra_dir2="outputs/track_a_dual_stream_long_lr2_${stamp2}"

echo "[wrapper] $(date -Is) starting Run 2: lr=${LR2}, motion_lr_ratio=${MOTION_LR_RATIO}"
echo "[wrapper] log:  ${log2}"
echo "[wrapper] ckpt: ${ckpt2}"

PYTHONPATH=src PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u -m smth2smth.pipelines.train \
    track=a \
    experiment="${EXPERIMENT}" \
    training.lr="${LR2}" \
    training.motion_lr_ratio="${MOTION_LR_RATIO}" \
    training.checkpoint_path="${REPO_ROOT}/${ckpt2}" \
    hydra.run.dir="${hydra_dir2}" \
    > "${log2}" 2>&1 < /dev/null &
child_pid2=$!
echo "${child_pid2}" > "${pid2}"
echo "[wrapper] Run 2 launched as PID ${child_pid2} (pidfile ${pid2})"

wait "${child_pid2}"
exit_code2=$?
echo "[wrapper] $(date -Is) Run 2 finished with exit code ${exit_code2}"
exit "${exit_code2}"
