# Round 2 — Diverse heads (stabilized) · single-machine launch runbook

**This file is the only doc you need on each GPU host** after `git pull` (configs ship in-repo; no local code or YAML edits).

**Background:** [`experiments/diver_CL_head_continue.md`](experiments/diver_CL_head_continue.md) · **Round 1 results:** [`experiments/diverse_CL_head_results.md`](experiments/diverse_CL_head_results.md)

---

## W&B — project, run names, tags (for panels)

| Field | Value |
|-------|--------|
| **Project** | `smth2smth-diverse-heads` (all Round 1 + Round 2 diverse-head trains) |
| **Entity** | From repo `.env` (`WANDB_ENTITY`) — same as other smth2smth runs |
| **Group** | **Not set** — the trainer does not pass `wandb.group`; use **project + tags** or **run name** in the UI |
| **Run display name** | `training.wandb.name` = same as `RUN_NAME` below (CLI override matches YAML) |

### Round 2 run names (one per machine)

| W&B run name | Architecture | Machine |
|--------------|--------------|---------|
| `arch2-perceiver-q16-stab` | Arch 2, Q=16 | baudroie |
| `arch3-divided-st-k3-stab` | Arch 3, K=3 | ablette |
| `arch3-divided-st-k6-stab` | Arch 3, K=6 | barbeau |
| `arch2-perceiver-q16-stab-s42` | Arch 2, Q=16, seed replica | gymnote (`seed=123` on CLI) |

### Tags (filter in W&B workspace)

Every stab run logs: `track_a`, `diverse_heads`, `stab`, plus arch tag (`arch2_perceiver` or `arch3_divided_st`). Seed replica adds `seed_replica`.

**Suggested panel filters**

- **Round 2 only:** project `smth2smth-diverse-heads` + tag `stab`
- **Compare to Round 1 honest val:** same project; run names without `-stab` (e.g. `arch2-perceiver-q16`) vs with `-stab`; chart `val/honest_top1` on stab runs and `val/top1` on Round 1
- **Holdout vs honest (Round 2):** `val/holdout_top1` (selection metric) vs `val/honest_top1` (monitoring; 90% of official val is in train)

Round 1 names for reference: `arch2-perceiver-q16`, `arch3-divided-st-k3`, `arch3-divided-st-k6`, `arch3-divided-st-k12`, `control-meanpool-ep500`.

**Project URL pattern:** `https://wandb.ai/<entity>/smth2smth-diverse-heads`

---

## What changed in Round 2 (no local tuning)

- Stabilized LLRD: backbone 5e-4 + LLRD; new modules `1e-4` + 10-epoch warmup; grad clip; stop if `mlp_activity_ratio > 10`
- **90/10 official val holdout** (`official_val_holdout_ratio: 0.1` in stab YAMLs): checkpoint/early-stop on holdout; full-val **honest** metric logged each epoch for W&B comparison
- Hydra presets: `track_a_diverse_arch2_perceiver_stab`, `track_a_diverse_arch3_divided_st_stab`, `track_a_diverse_arch3_divided_st_k3_stab`, `track_a_diverse_arch2_perceiver_stab_s42`

---

## Before launch (every machine, once)

```bash
cd /Data/romain.poggi/smth2smth
git pull   # needs Round-2 train.py + stab experiment YAMLs on main

# Required artifacts (not in git)
test -f checkpoints/track_a/ssl/videomaev2_t4native_encoder_ep500.pt && echo "SSL encoder OK" \
  || echo "MISSING encoder — rsync from a host that has it (e.g. brochet)"

test -f .env && echo "W&B .env OK" || echo "MISSING .env (WANDB_API_KEY)"

# Data paths: default Hydra dataset dirs (same as Round 1); fix only if this host uses different mounts

UV="$(command -v uv)"   # or full path, e.g. /users/.../.local/bin/uv
"$UV" run python -c "import torch; print('cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

**No control run** in this batch (ep500 mean-pool already done). **brochet / anguille** were skipped in the original plan (GPUs busy).

Optional smoke test: `./scripts/dryrun_diverse_heads_stab.sh`

---

## Launch — pick **one** block for your host

All blocks: repo root, `logs/track_a/`, W&B online, project `smth2smth-diverse-heads`. After ~90 s, `tail` the log (see § Health check).

### baudroie — `arch2-perceiver-q16-stab`

```bash
cd /Data/romain.poggi/smth2smth
UV="$(command -v uv)"
RUN_NAME=arch2-perceiver-q16-stab
EXPERIMENT=track_a_diverse_arch2_perceiver_stab
EXTRA=""
CKPT=checkpoints/track_a/videomaev2+ft/${RUN_NAME}.pt
SSL=checkpoints/track_a/ssl/videomaev2_t4native_encoder_ep500.pt
TAG=$(date +%Y%m%d)
LOG=logs/track_a/${RUN_NAME}_${TAG}.log
PID=logs/track_a/${RUN_NAME}_${TAG}.pid
mkdir -p logs/track_a checkpoints/track_a/videomaev2+ft
test -f "$SSL" || { echo "ABORT: missing $SSL"; exit 1; }
{
  echo "# run: ${RUN_NAME}"
  echo "# started: $(date -Iseconds)"
  echo "# track: a"
  echo "# experiment_doc: experiments/diver_CL_head_continue.md"
  echo "# hydra: experiment=${EXPERIMENT} ${EXTRA}"
  echo "# pid_file: ${PID}"
} > "$LOG"
PYTHONPATH=src PYTHONUNBUFFERED=1 nohup "$UV" run python -u -m smth2smth.pipelines.train \
  experiment=${EXPERIMENT} \
  model.init_from=${SSL} \
  training.checkpoint_path=${CKPT} \
  training.wandb.name=${RUN_NAME} \
  training.wandb.project=smth2smth-diverse-heads \
  ${EXTRA} >> "$LOG" 2>&1 &
echo $! > "$PID"
echo "launched ${RUN_NAME} pid $(cat "$PID") -> ${LOG}"
```

### ablette — `arch3-divided-st-k3-stab`

```bash
cd /Data/romain.poggi/smth2smth
UV="$(command -v uv)"
RUN_NAME=arch3-divided-st-k3-stab
EXPERIMENT=track_a_diverse_arch3_divided_st_k3_stab
EXTRA=""
CKPT=checkpoints/track_a/videomaev2+ft/${RUN_NAME}.pt
SSL=checkpoints/track_a/ssl/videomaev2_t4native_encoder_ep500.pt
TAG=$(date +%Y%m%d)
LOG=logs/track_a/${RUN_NAME}_${TAG}.log
PID=logs/track_a/${RUN_NAME}_${TAG}.pid
mkdir -p logs/track_a checkpoints/track_a/videomaev2+ft
test -f "$SSL" || { echo "ABORT: missing $SSL"; exit 1; }
{
  echo "# run: ${RUN_NAME}"
  echo "# started: $(date -Iseconds)"
  echo "# track: a"
  echo "# experiment_doc: experiments/diver_CL_head_continue.md"
  echo "# hydra: experiment=${EXPERIMENT} ${EXTRA}"
  echo "# pid_file: ${PID}"
} > "$LOG"
PYTHONPATH=src PYTHONUNBUFFERED=1 nohup "$UV" run python -u -m smth2smth.pipelines.train \
  experiment=${EXPERIMENT} \
  model.init_from=${SSL} \
  training.checkpoint_path=${CKPT} \
  training.wandb.name=${RUN_NAME} \
  training.wandb.project=smth2smth-diverse-heads \
  ${EXTRA} >> "$LOG" 2>&1 &
echo $! > "$PID"
echo "launched ${RUN_NAME} pid $(cat "$PID") -> ${LOG}"
```

### barbeau — `arch3-divided-st-k6-stab`

```bash
cd /Data/romain.poggi/smth2smth
UV="$(command -v uv)"
RUN_NAME=arch3-divided-st-k6-stab
EXPERIMENT=track_a_diverse_arch3_divided_st_stab
EXTRA=""
CKPT=checkpoints/track_a/videomaev2+ft/${RUN_NAME}.pt
SSL=checkpoints/track_a/ssl/videomaev2_t4native_encoder_ep500.pt
TAG=$(date +%Y%m%d)
LOG=logs/track_a/${RUN_NAME}_${TAG}.log
PID=logs/track_a/${RUN_NAME}_${TAG}.pid
mkdir -p logs/track_a checkpoints/track_a/videomaev2+ft
test -f "$SSL" || { echo "ABORT: missing $SSL"; exit 1; }
{
  echo "# run: ${RUN_NAME}"
  echo "# started: $(date -Iseconds)"
  echo "# track: a"
  echo "# experiment_doc: experiments/diver_CL_head_continue.md"
  echo "# hydra: experiment=${EXPERIMENT} ${EXTRA}"
  echo "# pid_file: ${PID}"
} > "$LOG"
PYTHONPATH=src PYTHONUNBUFFERED=1 nohup "$UV" run python -u -m smth2smth.pipelines.train \
  experiment=${EXPERIMENT} \
  model.init_from=${SSL} \
  training.checkpoint_path=${CKPT} \
  training.wandb.name=${RUN_NAME} \
  training.wandb.project=smth2smth-diverse-heads \
  ${EXTRA} >> "$LOG" 2>&1 &
echo $! > "$PID"
echo "launched ${RUN_NAME} pid $(cat "$PID") -> ${LOG}"
```

### gymnote — `arch2-perceiver-q16-stab-s42` (must pass `seed=123`)

```bash
cd /Data/romain.poggi/smth2smth
UV="$(command -v uv)"
RUN_NAME=arch2-perceiver-q16-stab-s42
EXPERIMENT=track_a_diverse_arch2_perceiver_stab_s42
EXTRA="seed=123"
CKPT=checkpoints/track_a/videomaev2+ft/${RUN_NAME}.pt
SSL=checkpoints/track_a/ssl/videomaev2_t4native_encoder_ep500.pt
TAG=$(date +%Y%m%d)
LOG=logs/track_a/${RUN_NAME}_${TAG}.log
PID=logs/track_a/${RUN_NAME}_${TAG}.pid
mkdir -p logs/track_a checkpoints/track_a/videomaev2+ft
test -f "$SSL" || { echo "ABORT: missing $SSL"; exit 1; }
{
  echo "# run: ${RUN_NAME}"
  echo "# started: $(date -Iseconds)"
  echo "# track: a"
  echo "# experiment_doc: experiments/diver_CL_head_continue.md"
  echo "# hydra: experiment=${EXPERIMENT} ${EXTRA}"
  echo "# pid_file: ${PID}"
} > "$LOG"
PYTHONPATH=src PYTHONUNBUFFERED=1 nohup "$UV" run python -u -m smth2smth.pipelines.train \
  experiment=${EXPERIMENT} \
  model.init_from=${SSL} \
  training.checkpoint_path=${CKPT} \
  training.wandb.name=${RUN_NAME} \
  training.wandb.project=smth2smth-diverse-heads \
  ${EXTRA} >> "$LOG" 2>&1 &
echo $! > "$PID"
echo "launched ${RUN_NAME} pid $(cat "$PID") -> ${LOG}"
```

**Monitor:** `tail -f "$LOG"` · `kill -0 $(cat "$PID")` · W&B URL in log line `[wandb] run started: https://...`

---

## Health check (~90 s after launch)

```bash
tail -n 50 "$LOG"
```

Expect: `[optim] stabilized LLRD:` (Arch 2/3), gradient clipping + MLP stop guard, Arch 3 `identity-at-init OK`, step lines every 25 steps, no traceback. After epoch 1: `val/holdout_top1` and `val/honest_top1` in W&B; `head/mlp_activity_ratio` on Arch 2/3.

---

## Machine run log (markdown)

After a healthy start, prepend to `report/$(whoami)/$(hostname | cut -d. -f1).md` per [distributed run log](.cursor/rules/distributed-run-log.mdc) (status, log link, W&B link, PID while running).
