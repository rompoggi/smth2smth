# Round 3 — Diverse heads (train-only comparison + seed replica) · single-machine launch runbook

**This file is the only doc you need on each GPU host** after `git pull` (configs ship in-repo; no local code or YAML edits — every run is the Round-2 stab recipe plus CLI overrides).

**Background:** [`experiments/diver_CL_head_continue.md`](experiments/diver_CL_head_continue.md) · **Round 2 launcher:** [`RUN_diverse_heads_stab.md`](RUN_diverse_heads_stab.md) · **Round 1 results:** [`experiments/diverse_CL_head_results.md`](experiments/diverse_CL_head_results.md)

---

## Why Round 3 (read first)

Round 2 (`-stab`) fixed the collapse: grad clip (`max_grad_norm=1.0`, `new_module_max_grad_norm=0.5`), `new_module_lr=1e-4` + 10-epoch warmup, and the `mlp_activity_ratio>10` stop-guard. Training is now smooth.

**The remaining problem is metric interpretation.** Round-2 runs use the 90/10 official-val holdout (`official_val_holdout_ratio=0.1`), so 90% of official val is in train:

| Metric (holdout-0.1 setup) | vs public LB | Why |
|----------------------------|-------------|-----|
| `holdout_top1` (n=676) | **≈ LB + 5 pp** | clean split, but small + slightly optimistic |
| `honest_top1` (n=6745) | **≈ LB + 10 pp** | 90% of these clips are in train (memorization) |

So neither holdout metric predicts LB well. **Round 3 runs the comparison grid train-only** (`official_val_holdout_ratio=0.0`): the model trains on `data/train` only and validates on the **full official val with no leakage**, whose `val/top1` tracks public LB within ~1 pp. That lets us **rank Q / K / architecture choices against an LB-faithful metric without spending Kaggle submissions.**

**One exception:** `arch2-perceiver-q16-stab-s7` stays on holdout 0.1 to line up with the s42 / s123 ensemble members already trained (these are the actual LB-submission models, trained on the full train + 90% val).

---

## What each run answers

- **Q-sweep (train-only):** Q=1 (`arch1-attn-probe`) · Q=8 · Q=16 · Q=32 — does query count help under full-FT MAE, measured on the LB-faithful metric?
- **K-sweep (train-only):** K=3 · K=6 · K=9 divided space-time temporal blocks — how many random-init temporal blocks a converged MAE backbone absorbs now that the head is stable (memory: K=12 fails the LLRD clash, K=9 is the ceiling).
- **Arch 4 (train-only):** AIM reused-MSA temporal adapters (orthogonal to Arch 3) — first measurement.
- **Holdout twins:** `*-q16-trainonly` / `*-k6-trainonly` / `*-k3-trainonly` are the train-only counterparts of the Round-2 holdout runs already going → quantifies the holdout→LB gap per architecture.
- **Seed replica (holdout):** `arch2-perceiver-q16-stab-s7` (seed=7) → third ensemble member alongside seed 42 / 123 for the Caruana stack.

---

## W&B — project, run names, tags

| Field | Value |
|-------|--------|
| **Project** | `smth2smth-diverse-heads` (all diverse-head trains, Rounds 1–3) |
| **Entity** | From repo `.env` (`WANDB_ENTITY`) |
| **Group** | Not set — filter by **project + run name** |
| **Run display name** | `training.wandb.name` = `RUN_NAME` below (CLI override) |

### Round 3 run names + machine assignment

| W&B run name | Architecture | Q / K | Data | Machine |
|--------------|--------------|-------|------|---------|
| `arch1-attn-probe-trainonly` | Arch 1 attn-probe | Q=1 | train-only | **lieu** |
| `arch2-perceiver-q8-trainonly` | Arch 2 Perceiver | Q=8 | train-only | **carrelet** |
| `arch2-perceiver-q16-trainonly` | Arch 2 Perceiver | Q=16 | train-only | **barbue** |
| `arch2-perceiver-q32-trainonly` | Arch 2 Perceiver | Q=32 | train-only | **labre** |
| `arch3-divided-st-k3-trainonly` | Arch 3 divided-ST | K=3 | train-only | **mulet** |
| `arch3-divided-st-k6-trainonly` | Arch 3 divided-ST | K=6 | train-only | **saumon** |
| `arch3-divided-st-k9-trainonly` | Arch 3 divided-ST | K=9 | train-only | **sole** (most VRAM) |
| `arch4-aim-trainonly` | Arch 4 AIM | 12 blocks | train-only | **thon** (most VRAM) |
| `arch2-perceiver-q16-stab-s7` | Arch 2 Perceiver | Q=16, seed=7 | **holdout 0.1** | **anguille** |

`sole` (K=9) and `thon` (Arch 4) have the most random-init temporal modules → most VRAM; if either host is tight, swap with a Q-sweep host.

### Tags

Runs built on a `*_stab` config inherit `track_a`, `diverse_heads`, `stab`, plus the arch tag. `arch1`/`arch4` runs (no stab config) carry only their base tags. Filter Round 3 by **run-name suffix**: `-trainonly` (the comparison grid) and `-s7` (the holdout replica).

**Suggested panels**

- **Train-only Q-sweep:** run names `arch1-attn-probe-trainonly`, `arch2-perceiver-q{8,16,32}-trainonly`; chart `val/top1` (honest, no leakage).
- **Train-only K-sweep:** `arch3-divided-st-k{3,6,9}-trainonly` + `arch4-aim-trainonly`; chart `val/top1`.
- **Holdout→LB gap:** `arch2-perceiver-q16-trainonly` `val/top1` vs `arch2-perceiver-q16-stab` `holdout_top1` (Round 2).
- **Seed ensemble:** `arch2-perceiver-q16-stab` / `-s42` / `-s7`; chart `holdout_top1`.

**Project URL:** `https://wandb.ai/<entity>/smth2smth-diverse-heads`

---

## What changed vs Round 2 (no local tuning)

- **Train-only for 8 of 9 runs** (`dataset.official_val_holdout_ratio=0.0`): validate on full official val, no leakage, `val/top1` ≈ public LB. `arch2-perceiver-q16-stab-s7` keeps holdout 0.1.
- **New axes:** Q ∈ {1, 8, 16, 32}; K ∈ {3, 6, 9}; Arch 1 and Arch 4 measured for the first time; seed=7 replica.
- **Stabilization recipe unchanged** (same as Round 2). `arch1`/`arch4` have no `*_stab` preset, so the recipe is passed explicitly on the CLI.
- **No new YAML** — every run is an existing `experiment=` preset plus `EXTRA` overrides.

**Unchanged:** ep500 SSL encoder, `tube_t=1` / `num_frames=4`, batch 8 × grad_accum 8, 50 epochs, cosine, EMA 0.9999, augment recipe, `log_interval_steps=25`, early-stop patience 15.

---

## Before launch (every machine, once)

```bash
cd /Data/romain.poggi/smth2smth
git pull   # needs Round-2 stab configs + train.py on main

# Required artifacts (not in git)
test -f checkpoints/track_a/ssl/videomaev2_t4native_encoder_ep500.pt && echo "SSL encoder OK" \
  || echo "MISSING encoder — rsync from a host that has it (e.g. brochet)"

test -f .env && echo "W&B .env OK" || echo "MISSING .env (WANDB_API_KEY)"

UV="$(command -v uv)"   # or full path, e.g. /users/.../.local/bin/uv
"$UV" run python -c "import torch; print('cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

Optional smoke test: `./scripts/dryrun_diverse_heads_stab.sh`

---

## Launch — pick the block for your host

All blocks: repo root, `logs/track_a/`, W&B online, project `smth2smth-diverse-heads`. After ~90 s, `tail` the log (see § Health check). Each block carries its overrides in `EXTRA`; `RUN_NAME` drives the log path, checkpoint path, and W&B name.

### lieu — `arch1-attn-probe-trainonly` (Q=1)

```bash
cd /Data/romain.poggi/smth2smth
UV="$(command -v uv)"
RUN_NAME=arch1-attn-probe-trainonly
EXPERIMENT=track_a_diverse_arch1_attn_probe
EXTRA="dataset.official_val_holdout_ratio=0.0 training.new_module_lr=1e-4 training.new_module_warmup_epochs=10 training.max_grad_norm=1.0 training.new_module_max_grad_norm=0.5 training.stop_on_mlp_activity_ratio=10.0"
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

### carrelet — `arch2-perceiver-q8-trainonly`

```bash
cd /Data/romain.poggi/smth2smth
UV="$(command -v uv)"
RUN_NAME=arch2-perceiver-q8-trainonly
EXPERIMENT=track_a_diverse_arch2_perceiver_stab
EXTRA="model.head_queries=8 dataset.official_val_holdout_ratio=0.0"
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

### barbue — `arch2-perceiver-q16-trainonly`

```bash
cd /Data/romain.poggi/smth2smth
UV="$(command -v uv)"
RUN_NAME=arch2-perceiver-q16-trainonly
EXPERIMENT=track_a_diverse_arch2_perceiver_stab
EXTRA="dataset.official_val_holdout_ratio=0.0"
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

### labre — `arch2-perceiver-q32-trainonly`

```bash
cd /Data/romain.poggi/smth2smth
UV="$(command -v uv)"
RUN_NAME=arch2-perceiver-q32-trainonly
EXPERIMENT=track_a_diverse_arch2_perceiver_stab
EXTRA="model.head_queries=32 dataset.official_val_holdout_ratio=0.0"
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

### mulet — `arch3-divided-st-k3-trainonly`

```bash
cd /Data/romain.poggi/smth2smth
UV="$(command -v uv)"
RUN_NAME=arch3-divided-st-k3-trainonly
EXPERIMENT=track_a_diverse_arch3_divided_st_k3_stab
EXTRA="dataset.official_val_holdout_ratio=0.0"
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

### saumon — `arch3-divided-st-k6-trainonly`

```bash
cd /Data/romain.poggi/smth2smth
UV="$(command -v uv)"
RUN_NAME=arch3-divided-st-k6-trainonly
EXPERIMENT=track_a_diverse_arch3_divided_st_stab
EXTRA="dataset.official_val_holdout_ratio=0.0"
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

### sole — `arch3-divided-st-k9-trainonly` (most VRAM)

```bash
cd /Data/romain.poggi/smth2smth
UV="$(command -v uv)"
RUN_NAME=arch3-divided-st-k9-trainonly
EXPERIMENT=track_a_diverse_arch3_divided_st_stab
EXTRA="model.temporal_layers=9 dataset.official_val_holdout_ratio=0.0"
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

### thon — `arch4-aim-trainonly` (most VRAM)

```bash
cd /Data/romain.poggi/smth2smth
UV="$(command -v uv)"
RUN_NAME=arch4-aim-trainonly
EXPERIMENT=track_a_diverse_arch4_aim
EXTRA="dataset.official_val_holdout_ratio=0.0 training.new_module_lr=1e-4 training.new_module_warmup_epochs=10 training.max_grad_norm=1.0 training.new_module_max_grad_norm=0.5 training.stop_on_mlp_activity_ratio=10.0"
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

### anguille — `arch2-perceiver-q16-stab-s7` (holdout 0.1, seed=7)

```bash
cd /Data/romain.poggi/smth2smth
UV="$(command -v uv)"
RUN_NAME=arch2-perceiver-q16-stab-s7
EXPERIMENT=track_a_diverse_arch2_perceiver_stab
EXTRA="seed=7"
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

Expect:

- `[optim] stabilized LLRD:` (all runs route the head — and temporal modules on Arch 3/4 — to the new-module group).
- **Train-only runs:** `[data] use_official_val=true: train=44993 ...` and the per-epoch val line labeled **`val`** (honest, n=6745), **not** `holdout`. If you see `val_holdout=…`, the `official_val_holdout_ratio=0.0` override did not take — stop and check `EXTRA`.
- **`arch2-perceiver-q16-stab-s7`:** `[data] ... val_holdout=676 / val_total=6745` and `seed=7` echoed.
- Arch 3 (`k3/k6/k9`) and Arch 4: `[diverse-arch] ... identity-at-init OK`.
- Step lines every 25 steps, no traceback. First-epoch `head/mlp_activity_ratio` should be `< 2`; the run self-stops if it ever exceeds 10.

If broken: stop, report, relaunch into the **same** log path after the fix.

---

## Machine run log (markdown)

After a healthy start, prepend to `report/$(whoami)/$(hostname | cut -d. -f1).md` per [distributed run log](.claude/rules/distributed-run-log.md): status, log link, W&B link, PID while running. **Restart/VM resume:** append to the **same** log file and reuse the same `RUN_NAME` (`WANDB_RESUME=allow` + `WANDB_RUN_ID=<id>`).
