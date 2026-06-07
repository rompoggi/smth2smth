#!/usr/bin/env bash
# Collect seed-42 arch2-perceiver-q*-trainonly runs -> perceiverQ*-mae500-s42 on coordinator.
set -euo pipefail

REPO="${REPO:-/Data/romain.poggi/smth2smth}"
cd "$REPO"
DEST_CKPT="${REPO}/checkpoints/track_a/perceiver_q_s42"
DEST_LOG="${REPO}/logs/track_a/perceiver_q_s42"
mkdir -p "$DEST_CKPT" "$DEST_LOG"

# Q -> old_run:new_run:ckpt_source:log_source
# ckpt_source: host:path or local:path
# log_source: host:path | wandb:run_id | local:path
declare -A SPECS=(
  [2]="arch2-perceiver-q2-trainonly:perceiverQ2-mae500-s42:brochet:${REPO}/checkpoints/track_a/videomaev2+ft/arch2-perceiver-q2-trainonly.pt:brochet:${REPO}/logs/track_a/arch2-perceiver-q2-trainonly_20260526.log"
  [4]="arch2-perceiver-q4-trainonly:perceiverQ4-mae500-s42:lieu:${REPO}/checkpoints/track_a/videomaev2+ft/arch2-perceiver-q4-trainonly.pt:wandb:yfms3io3"
  [8]="arch2-perceiver-q8-trainonly:perceiverQ8-mae500-s42:local:${REPO}/checkpoints/track_a/round3_collected/arch2-perceiver-q8-trainonly.final-ep50.pt:wandb:mbb75n3k"
  [16]="arch2-perceiver-q16-trainonly:perceiverQ16-mae500-s42:local:${REPO}/checkpoints/track_a/round3_collected/arch2-perceiver-q16-trainonly.final-ep50.pt:wandb:cfqjfqy1"
  [32]="arch2-perceiver-q32-trainonly:perceiverQ32-mae500-s42:local:${REPO}/checkpoints/track_a/round3_collected/arch2-perceiver-q32-trainonly.final-ep50.pt:wandb:18la2ij5"
)

here="$(hostname -s 2>/dev/null || hostname)"
MANIFEST="${DEST_LOG}/manifest.txt"
: >"$MANIFEST"
{
  echo "# collected: $(date -Is)"
  echo "# seed: 42 | train-only (include_val_in_train=false) | SSL ep500 | Perceiver head"
  echo "# renamed: arch2-perceiver-q*-trainonly -> perceiverQ*-mae500-s42"
  echo "# source W&B project: smth2smth-diverse-heads"
  echo "# columns: Q old_run new_run log ckpt status best_val note"
} >>"$MANIFEST"

fetch_wandb_output() {
  local run_id="$1" dest="$2"
  if ! PYTHONPATH=src uv run python - <<PY
import wandb
from pathlib import Path
api = wandb.Api()
r = api.run("romain-poggi-ecole-polytechnique/smth2smth-diverse-heads/${run_id}")
names = {f.name for f in r.files()}
if "output.log" not in names:
    raise SystemExit("NO_OUTPUT_LOG")
tmp = Path("${dest}.tmp")
tmp.parent.mkdir(parents=True, exist_ok=True)
r.file("output.log").download(root=str(tmp.parent), replace=True)
(Path(tmp.parent) / "output.log").rename(tmp)
print(tmp)
PY
  then
    cat >"$dest" <<EOF
# no local or wandb output.log for run_id=${run_id}
# metrics replay uses W&B history via upload_perceiver_q_s42_wandb.py --q=${run_id}
EOF
    echo "WARN: no output.log for wandb run ${run_id}; wrote stub ${dest}" >&2
  fi
  [ -f "${dest}.tmp" ] && mv "${dest}.tmp" "$dest" || true
}

copy_ckpt() {
  local src_spec="$1" dest="$2"
  IFS=: read -r kind path <<< "$src_spec"
  if [[ "$kind" == local ]]; then
    cp -a "$path" "$dest"
  elif [[ "$kind" == "$here" ]]; then
    cp -a "$path" "$dest"
  else
    rsync -az "${kind}:${path}" "$dest"
  fi
}

for Q in 2 4 8 16 32; do
  IFS=: read -r OLD NEW ckpt_host ckpt_path log_kind log_ref <<< "${SPECS[$Q]}"
  echo "=== COLLECT Q${Q} $OLD -> $NEW ==="

  LOG_DEST="${DEST_LOG}/${NEW}_20260526.log"
  CKPT_DEST="${DEST_CKPT}/${NEW}.pt"

  copy_ckpt "${ckpt_host}:${ckpt_path}" "$CKPT_DEST"

  if [[ "$log_kind" == wandb ]]; then
    fetch_wandb_output "$log_ref" "$LOG_DEST"
    echo "# log_source: wandb output.log run_id=${log_ref}" >>"$LOG_DEST"
  elif [[ "$log_kind" == local ]]; then
    cp -a "$log_ref" "$LOG_DEST"
  elif [[ "$log_kind" == "$here" ]]; then
    cp -a "$log_ref" "$LOG_DEST"
  else
    rsync -az "${log_kind}:${log_ref}" "$LOG_DEST"
  fi

  META="$(python3 - <<META
import re
from pathlib import Path
p = Path("$LOG_DEST")
text = p.read_text(errors="replace")
done_honest = re.findall(r"Done\. Best val honest top1: ([\d.]+)", text)
done_plain = re.findall(r"Done\. Best val top1: ([\d.]+)", text)
best = (done_honest or done_plain or [""])[-1]
epochs = [int(m.group(1)) for m in re.finditer(r"Epoch (\d+)/50", text)]
last_ep = max(epochs) if epochs else 0
if best and last_ep >= 50:
    status = "DONE"
elif best:
    status = "DONE"
elif last_ep < 50:
    status = "INCOMPLETE"
else:
    status = "UNKNOWN"
note = ""
if "$log_kind" == "wandb" and "$Q" == "4":
    note = "log from wandb output (no local log); original run state=crashed ep49 val in cloud"
print(f"{status}|{best}|{note}")
META
)"

  IFS='|' read -r status best_val note <<< "$META"
  echo "$Q $OLD $NEW $(basename "$LOG_DEST") $(basename "$CKPT_DEST") $status ${best_val:-n/a} ${note}" >>"$MANIFEST"
done

echo "Collected to:"
echo "  $DEST_CKPT"
echo "  $DEST_LOG"
echo "  $MANIFEST"
