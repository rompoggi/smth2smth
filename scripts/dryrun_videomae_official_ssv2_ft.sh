#!/usr/bin/env bash
# 1-epoch smoke test for the official SSv2 VideoMAE FT recipe.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSL_ENCODER="${1:-${REPO_ROOT}/checkpoints/track_a/ssl/espadon_t16_encoder_ep50.pt}"
RUN_NAME="${RUN_NAME:-mae50-ft-f16-dryrun}"

export SSL_ENCODER
export RUN_NAME
export NOHUP=0
export TAG="${TAG:-dryrun}"
export EXPERIMENT=track_a_videomae_official_ssv2_ft_dryrun
export FT_CHECKPOINT="${REPO_ROOT}/checkpoints/track_a/videomaev2+ft/_dryrun_${RUN_NAME}.pt"

exec "${REPO_ROOT}/scripts/launch_videomae_official_ssv2_ft.sh"
