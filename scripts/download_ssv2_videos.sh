#!/usr/bin/env bash
# Download SSv2 video parts incrementally (Qualcomm software center).
#
# Agentic workflow (recommended):
#   1. Download ONE part:  ./scripts/download_ssv2_videos.sh 00
#   2. Analyze coverage:   PYTHONPATH=src .venv/bin/python scripts/analyze_ssv2_parts.py \\
#        --part-archive data/ssv2/raw/archives/20bn-something-something-v2-00
#   3. Extract if useful:  ./scripts/download_ssv2_videos.sh extract 00
#   4. Repeat for parts 01, 02, … only when you need more clip volume.
#
# Annotations (no registration): scripts/build_extended_train.py download-annotations
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCH="${SSV2_ARCHIVES:-$REPO_ROOT/data/ssv2/raw/archives}"
OUT="${SSV2_VIDEOS:-$REPO_ROOT/data/ssv2/raw/20bn-something-something-v2}"
BASE_URL="https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2"

download_part() {
  local part="$1"
  local dest="$ARCH/20bn-something-something-v2-${part}"
  mkdir -p "$ARCH"
  echo "Downloading part ${part} → ${dest}"
  curl -fL -C - -o "$dest" "${BASE_URL}/20bn-something-something-v2-${part}"
}

extract_part() {
  local part="$1"
  local src="$ARCH/20bn-something-something-v2-${part}"
  mkdir -p "$OUT"
  echo "Extracting ${src} → ${OUT}/"
  tar -xzf "$src" -C "$OUT" --strip-components=1 2>/dev/null || tar -xzf "$src" -C "$(dirname "$OUT")"
}

cmd="${1:-help}"
case "$cmd" in
  [0-9]|[0-1][0-9])
    download_part "$(printf '%02d' "$1")"
    ;;
  extract)
    extract_part "$(printf '%02d' "$2")"
    ;;
  all)
    for p in $(seq -w 0 19); do download_part "$p"; done
    ;;
  *)
    echo "Usage: $0 <00-19> | extract <00-19> | all"
    exit 1
    ;;
esac
