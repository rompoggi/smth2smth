#!/usr/bin/env bash
# Pull seed-43 FT MAE scaling logs + checkpoints from fleet hosts to gymnote (coordinator).
set -euo pipefail

REPO="${REPO:-/Data/romain.poggi/smth2smth}"
cd "$REPO"
DEST_CKPT="${REPO}/checkpoints/track_a/ft_mae_scaling_s43"
DEST_LOG="${REPO}/logs/track_a/ft_mae_scaling_s43"
mkdir -p "$DEST_CKPT" "$DEST_LOG"

# host -> run (original s43 fleet)
declare -A HOST_RUN=(
  [gardon]=meanpool-mae050-s43
  [gymnote]=meanpool-mae100-s43
  [labre]=meanpool-mae150-s43
  [lieu]=meanpool-mae200-s43
  [lotte]=meanpool-mae250-s43
  [mulet]=meanpool-mae300-s43
  [murene]=meanpool-mae350-s43
  [piranha]=meanpool-mae400-s43
  [raie]=meanpool-mae450-s43
  [requin]=meanpool-mae500-s43
  [rouget]=perceiverQ16-mae100-s43
  [roussette]=perceiverQ16-mae200-s43
  [sole]=perceiverQ16-mae300-s43
  [thon]=perceiverQ16-mae400-s43
  [truite]=perceiverQ16-mae500-s43
)

MANIFEST="${DEST_LOG}/manifest.txt"
: >"$MANIFEST"
echo "# collected: $(date -Is)" >>"$MANIFEST"
echo "# host run log ckpt best_val" >>"$MANIFEST"

here="$(hostname -s 2>/dev/null || hostname)"
for H in "${!HOST_RUN[@]}"; do
  RUN="${HOST_RUN[$H]}"
  echo "=== COLLECT $H -> $RUN ==="
  if [[ "$H" == "$here" ]]; then
    LOG_SRC="$(ls -t "${REPO}/logs/track_a/${RUN}"_*.log 2>/dev/null | head -1 || true)"
    CKPT_SRC="${REPO}/checkpoints/track_a/videomaev2+ft/${RUN}.pt"
  else
    LOG_SRC="$(ssh -o BatchMode=yes "$H" "ls -t ${REPO}/logs/track_a/${RUN}_*.log 2>/dev/null | head -1" || true)"
    CKPT_SRC="${H}:${REPO}/checkpoints/track_a/videomaev2+ft/${RUN}.pt"
  fi
  if [[ -z "$LOG_SRC" ]]; then
    echo "WARN: no log for $RUN on $H" >&2
    continue
  fi
  LOG_BASE="$(basename "$LOG_SRC")"
  if [[ "$H" == "$here" ]]; then
    cp -a "$LOG_SRC" "${DEST_LOG}/${LOG_BASE}"
    cp -a "$CKPT_SRC" "${DEST_CKPT}/${RUN}.pt" 2>/dev/null || echo "WARN: missing ckpt $RUN on $H"
  else
    rsync -az "${H}:${LOG_SRC}" "${DEST_LOG}/${LOG_BASE}"
    rsync -az "$CKPT_SRC" "${DEST_CKPT}/${RUN}.pt" || echo "WARN: missing ckpt $RUN on $H"
  fi
  BEST="$(grep 'Done\. Best' "${DEST_LOG}/${LOG_BASE}" 2>/dev/null | tail -1 || true)"
  echo "$H $RUN ${LOG_BASE} ${RUN}.pt ${BEST}" >>"$MANIFEST"
done

echo "Collected to:"
echo "  $DEST_CKPT"
echo "  $DEST_LOG"
echo "  $MANIFEST"
