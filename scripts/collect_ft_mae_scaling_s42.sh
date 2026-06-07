#!/usr/bin/env bash
# Collect seed-42 maeEEE-ft-f4 mean-pool scaling runs → meanpool-maeXXX-s42 on coordinator.
set -euo pipefail

REPO="${REPO:-/Data/romain.poggi/smth2smth}"
cd "$REPO"
DEST_CKPT="${REPO}/checkpoints/track_a/ft_mae_scaling_s42"
DEST_LOG="${REPO}/logs/track_a/ft_mae_scaling_s42"
mkdir -p "$DEST_CKPT" "$DEST_LOG"

# Authoritative host for each SSL pretrain epoch (seed 42, honest val, T=4).
declare -A HOST_RUN=(
  [50]=truite:mae50-ft-f4
  [100]=lieu:mae100-ft-f4
  [150]=rouget:mae150-ft-f4
  [200]=roussette:mae200-ft-f4
  [250]=labre:mae250-ft-f4
  [300]=roussette:mae300-ft-f4
  [350]=rouget:mae350-ft-f4
  [400]=lieu:mae400-ft-f4
  [450]=labre:mae450-ft-f4
  [500]=sole:mae500-ft-f4
)

find_train_log() {
  local H="$1" OLD="$2"
  ssh -o BatchMode=yes "$H" bash -s -- "$REPO" "$OLD" <<'FIND'
REPO="$1" OLD="$2"
for d in "$REPO/logs/track_a" "$REPO/logs"; do
  for f in "$d/${OLD}"_*.log; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    [[ "$base" =~ ^${OLD}_[0-9]{8}\.log$ ]] || continue
    echo "$f"
    exit 0
  done
done
FIND
}

here="$(hostname -s 2>/dev/null || hostname)"
MANIFEST="${DEST_LOG}/manifest.txt"
: >"$MANIFEST"
{
  echo "# collected: $(date -Is)"
  echo "# seed: 42 | recipe: track_a_videomae_official_ssv2_ft T=4 honest val"
  echo "# renamed: maeEEE-ft-f4 -> meanpool-maeEEE-s42"
  echo "# columns: ssl_ep host old_run new_run log ckpt status best_val wandb_merge note"
} >>"$MANIFEST"

for ep in 50 100 150 200 250 300 350 400 450 500; do
  IFS=: read -r H OLD <<< "${HOST_RUN[$ep]}"
  NEW=$(printf "meanpool-mae%03d-s42" "$ep")
  echo "=== COLLECT ep${ep} $H $OLD -> $NEW ==="

  LOG_SRC="$(find_train_log "$H" "$OLD")"
  if [[ -z "$LOG_SRC" ]]; then
    echo "WARN: no training log for $OLD on $H" >&2
    echo "$ep $H $OLD $NEW - - MISSING_LOG - -" >>"$MANIFEST"
    continue
  fi

  CKPT_SRC="${REPO}/checkpoints/track_a/videomaev2+ft/${OLD}.pt"
  LOG_BASE="${NEW}_$(basename "$LOG_SRC" | sed "s/^${OLD}_//")"
  LOG_DEST="${DEST_LOG}/${LOG_BASE}"
  CKPT_DEST="${DEST_CKPT}/${NEW}.pt"

  if [[ "$H" == "$here" ]]; then
    cp -a "$LOG_SRC" "$LOG_DEST"
    if [[ -f "$CKPT_SRC" ]]; then cp -a "$CKPT_SRC" "$CKPT_DEST"; else echo "WARN: no ckpt $CKPT_SRC"; fi
  else
    rsync -az "${H}:${LOG_SRC}" "$LOG_DEST"
    rsync -az "${H}:${CKPT_SRC}" "$CKPT_DEST" || echo "WARN: missing ckpt $OLD on $H"
  fi

  META="$(ssh -o BatchMode=yes "$H" bash -s -- "$LOG_SRC" "$CKPT_SRC" <<'META'
LOG="$1" CKPT="$2"
done=$(grep 'Done\. Best val top1:' "$LOG" 2>/dev/null | tail -1 | grep -oE '0\.[0-9]+' | tail -1)
last_ep=$(grep -oE 'Epoch [0-9]+/50' "$LOG" 2>/dev/null | tail -1 | awk '{print $2}' | cut -d/ -f1)
wandb_n=$(grep -c '\[wandb\] run started:' "$LOG" 2>/dev/null || echo 0)
wandb_ids=$(grep '\[wandb\] run started:' "$LOG" 2>/dev/null | sed 's/.*runs\///' | sort -u | tr '\n' ',' | sed 's/,$//')
resume=$(grep -cE 'VM restart|=== RESUME|resume_from' "$LOG" 2>/dev/null || echo 0)
ckpt_ok=$([ -f "$CKPT" ] && echo yes || echo no)
if [[ -n "$done" && "$last_ep" == "50" ]]; then status=DONE
elif [[ -n "$done" ]]; then status=DONE
elif [[ "${last_ep:-0}" -lt 50 ]]; then status=INCOMPLETE
else status=UNKNOWN
fi
merge=no
[[ "$wandb_n" -gt 1 ]] && merge=yes
note=""
[[ "$status" == INCOMPLETE ]] && note="stopped ep${last_ep:-?}; ckpt=${ckpt_ok}"
printf "%s|%s|%s|%s|%s\n" "$status" "${done:-}" "$merge" "$wandb_ids" "$note"
META
)"

  IFS='|' read -r status best_val wandb_merge wandb_ids note <<< "$META"
  echo "$ep $H $OLD $NEW ${LOG_BASE} ${NEW}.pt $status ${best_val:-n/a} wandb_merge=${wandb_merge} ids=${wandb_ids} ${note}" >>"$MANIFEST"
done

echo "Collected to:"
echo "  $DEST_CKPT"
echo "  $DEST_LOG"
echo "  $MANIFEST"
