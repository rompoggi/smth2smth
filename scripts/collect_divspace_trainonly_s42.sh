#!/usr/bin/env bash
# Collect arch3-divided-st-k{1,3,6,9,12}-trainonly -> DivSpaceTimeK*-mae500-s42 on coordinator.
set -euo pipefail

REPO="${REPO:-/Data/romain.poggi/smth2smth}"
cd "$REPO"
DEST_CKPT="${REPO}/checkpoints/track_a/divspace_s42"
DEST_LOG="${REPO}/logs/track_a/divspace_s42"
mkdir -p "$DEST_CKPT" "$DEST_LOG"

# old_run:new_run:host:old_ckpt_base
SPECS=(
  "arch3-divided-st-k3-trainonly:DivSpaceTimeK3-mae500-s42:barbeau"
  "arch3-divided-st-k6-trainonly:DivSpaceTimeK6-mae500-s42:saumon"
  "arch3-divided-st-k9-trainonly:DivSpaceTimeK9-mae500-s42:sole"
)

MANIFEST="${DEST_LOG}/manifest.txt"
{
  echo "# collected: $(date -Is)"
  echo "# seed: 42 | train-only | SSL ep500 | divided space-time (arch3)"
  echo "# rename: arch3-divided-st-k*-trainonly -> DivSpaceTimeK*-mae500-s42"
  echo "# columns: K old_run new_run host ckpt log wandb_id note"
} >"$MANIFEST"

for spec in "${SPECS[@]}"; do
  IFS=: read -r OLD NEW HOST <<<"$spec"
  K="${NEW#DivSpaceTimeK}"
  K="${K%%-*}"
  SRC_PT="${REPO}/checkpoints/track_a/videomaev2+ft/${OLD}.pt"
  SRC_LAST="${REPO}/checkpoints/track_a/videomaev2+ft/${OLD}.last.pt"
  LOG_SRC="${REPO}/logs/track_a/${OLD}_20260526.log"
  echo "=== $HOST $OLD -> $NEW ==="
  for f in "$SRC_PT" "$SRC_LAST"; do
    ssh -o BatchMode=yes "$HOST" "test -f '$f'" || { echo "MISSING $f on $HOST" >&2; exit 1; }
  done
  rsync -az "${HOST}:${SRC_PT}" "${DEST_CKPT}/${NEW}.pt"
  rsync -az "${HOST}:${SRC_LAST}" "${DEST_CKPT}/${NEW}.last.pt"
  rsync -az "${HOST}:${LOG_SRC}" "${DEST_LOG}/${NEW}_20260526.log" 2>/dev/null || true
  WID=$(ssh -o BatchMode=yes "$HOST" "grep -oP 'runs/\\K[a-z0-9]+$' '$LOG_SRC' | head -1" 2>/dev/null || true)
  echo "${K} ${OLD} ${NEW} ${HOST} ${DEST_CKPT}/${NEW}.pt ${DEST_LOG}/${NEW}_20260526.log ${WID}" >>"$MANIFEST"
  ls -lh "${DEST_CKPT}/${NEW}.pt" "${DEST_CKPT}/${NEW}.last.pt"
done

echo "Done. Manifest: $MANIFEST"
