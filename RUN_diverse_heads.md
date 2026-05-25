# Overnight runbook — Diverse classifier heads (Arch 1–4) on 8× RTX 3090

**Goal:** each machine fine-tunes **one** VideoMAE ViT-B variant on the 33-class SSv2
subset, all from the **same** SSL encoder, so the results are directly comparable and
can be Caruana-stacked afterwards. Spec: [`experiments/diverse_classifier_heads_post_mae.md`](experiments/diverse_classifier_heads_post_mae.md).

**One machine = one experiment.** Find your hostname in the table, copy the launch
block, verify health, log it. Each run is ~50 epochs, fits in 24 GB, finishes well
within a night.

---

## 0. Assignment table

| Machine    | `RUN_NAME`                         | `EXPERIMENT`                          | `EXTRA` (Hydra overrides)              | What it is |
|------------|------------------------------------|---------------------------------------|----------------------------------------|------------|
| **raie**     | `control-meanpool`      | `track_a_videomae_official_ssv2_ft`   | `model.tube_t=1 dataset.num_frames=4`  | Control: mean-pool head (fair baseline on ep500) |
| **piranha**  | `arch1-attn-probe`      | `track_a_diverse_arch1_attn_probe`    | *(none)*                               | Arch 1: single-query attentive probe |
| **lotte**    | `arch2-perceiver-q16`   | `track_a_diverse_arch2_perceiver`     | *(none)*                               | Arch 2: 16-query Perceiver |
| **brochet**  | `arch3-divided-st-k6`   | `track_a_diverse_arch3_divided_st`    | *(none)*                               | Arch 3: divided space-time, last K=6 (highest EV) |
| **murene**   | `arch4-aim`             | `track_a_diverse_arch4_aim`           | *(none)*                               | Arch 4: AIM reused-MSA, all 12 blocks |
| **ablette**  | `arch3-divided-st-k9`   | `track_a_diverse_arch3_divided_st`    | `model.temporal_layers=9`              | Arch 3 variant: K=9 (K-sweep) |
| **anguille** | `arch3-divided-st-k12`  | `track_a_diverse_arch3_divided_st`    | `model.temporal_layers=12`             | Arch 3 variant: K=12 (K-sweep) |
| **sole**     | `arch2-perceiver-q32`   | `track_a_diverse_arch2_perceiver`     | `model.head_queries=32`                | Arch 2 variant: Q=32 (Q-sweep) |

The 5 core runs are raie–murene. The 3 spares (ablette/anguille/sole) front-load the
report's recommended follow-ups: the **K-sweep {6,9,12}** for Arch 3 (the report
expects K\*=9–12 on SSv2) and one extra **Q point** for Arch 2. Reassign freely — any
machine can run any row; only `RUN_NAME`/`EXPERIMENT`/`EXTRA` change.

---

## 1. Prerequisites (once per machine, before launching)

```bash
cd /Data/romain.poggi/smth2smth

# 1a. Get the diverse-heads code + configs. They live on origin/main; this machine
#     must have the commit that adds configs/experiment/track_a_diverse_*.yaml and
#     the CrossAttnPoolHead / DividedSpaceTimeBlock / AIMReuseBlock model code.
git pull --ff-only origin main
git log --oneline -1            # confirm you have the diverse-heads commit

# 1b. Confirm the experiment configs are present (should list 4 files).
ls configs/experiment/track_a_diverse_*.yaml

# 1c. Confirm the SSL encoder is present LOCALLY. /Data is a local NVMe disk on each
#     machine (NOT shared), and this 345 MB checkpoint is gitignored — so it must be
#     copied to every machine separately.
test -f checkpoints/track_a/ssl/videomaev2_t4native_encoder_ep500.pt \
  && echo "SSL encoder OK" \
  || echo "MISSING — copy it here, e.g.:  rsync -avP <host_with_it>:/Data/romain.poggi/smth2smth/checkpoints/track_a/ssl/videomaev2_t4native_encoder_ep500.pt checkpoints/track_a/ssl/"

# 1d. Confirm the GPU is free and uv works.
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
uv run python -c "import torch; print('cuda', torch.cuda.is_available())"
```

**W&B:** `train.py` auto-loads the repo `.env` (for `WANDB_API_KEY`). If this machine
has no `.env`, either copy one over **or** prepend `WANDB_MODE=offline` to the launch
command (training is unaffected; you just won't get a live dashboard).

---

## 2. Launch (copy-paste; edit only the three lines in the box)

```bash
cd /Data/romain.poggi/smth2smth

# ╔════════════ EDIT THESE FOR YOUR MACHINE (from the table) ════════════╗
RUN_NAME=arch3-divided-st-k6
EXPERIMENT=track_a_diverse_arch3_divided_st
EXTRA=""          # e.g. "model.temporal_layers=9"  or  "model.head_queries=32"
# ╚══════════════════════════════════════════════════════════════════════╝

SSL=checkpoints/track_a/ssl/videomaev2_t4native_encoder_ep500.pt
TAG=$(date +%Y%m%d)
LOG=logs/track_a/${RUN_NAME}_${TAG}.log
PID=logs/track_a/${RUN_NAME}_${TAG}.pid
CKPT=checkpoints/track_a/videomaev2+ft/${RUN_NAME}.pt
mkdir -p logs/track_a checkpoints/track_a/videomaev2+ft

test -f "$SSL" || { echo "ABORT: missing $SSL (see step 1c)"; }

# ASCII run-log header (per start-resume-runs.md), then nohup-append the trainer.
{
  echo "# run: ${RUN_NAME}"
  echo "# started: $(date -Iseconds)"
  echo "# track: a"
  echo "# experiment_doc: experiments/diverse_classifier_heads_post_mae.md"
  echo "# hydra: experiment=${EXPERIMENT} ${EXTRA}"
} > "$LOG"

PYTHONPATH=src PYTHONUNBUFFERED=1 nohup uv run python -u -m smth2smth.pipelines.train \
  experiment=${EXPERIMENT} \
  model.init_from=${SSL} \
  training.checkpoint_path=${CKPT} \
  training.wandb.name=${RUN_NAME} \
  training.wandb.project=smth2smth-diverse-heads \
  ${EXTRA} >> "$LOG" 2>&1 &

echo $! > "$PID"
echo "launched ${RUN_NAME} (pid $(cat "$PID")) -> ${LOG}"
```

Notes:
- `RUN_NAME` drives the checkpoint name, log name, and W&B name, so variants never
  collide. The `experiment=` config already bakes `tube_t=1`, `num_frames=4`, the head,
  and (for Arch 3/4) the temporal settings; the control row needs the two overrides in
  its `EXTRA` because that preset defaults to 16 frames.
- All 8 land in W&B project **`smth2smth-diverse-heads`** for side-by-side comparison.

---

## 3. Health check (~60–90 s after launch)

```bash
tail -n 40 "$LOG"          # (use the same $LOG, or logs/track_a/<RUN_NAME>_<date>.log)
```

A healthy start shows, in order:
1. The composed config (no `ConfigCompositionException`, no `ConstructorError`).
2. `[init_from] loaded 149 encoder tensors ... encoder-missing=N` — `N=0` for
   control/Arch 1/Arch 2; `N=48` for Arch 3 (K=6) / `72` (K=9) / `96` (K=12);
   `N=120` for Arch 4. These are the new temporal tensors and are **expected** to be
   "missing" from the SSL checkpoint (they're freshly initialised).
3. **Arch 3 / Arch 4 only — the single highest-payoff check:**
   `[diverse-arch] temporal_mode=... ; identity-at-init OK (temporal path == identity at step 0)`.
   If it ever says **BROKEN**, stop the run and report — the temporal blocks would
   train from scratch and lose 1–2+ pt.
4. `[diverse-arch] pool_head: num_queries=... init query pairwise cosine≈0.0x` (Arch 1/2/3/4).
5. `[optim] LLRD: layer_decay=0.75, depth=12, 28 param groups` and
   `[optim] trainable ... (100.00%)`.
6. Training step lines at `log_interval_steps=25`, one per line, ASCII only, **no
   `Traceback`**.

If the log only has config + a Python traceback, fix the cause, then relaunch into the
**same** `$LOG` path.

Per-epoch you will also see (Arch 1/2/3/4): `[head-diag] mlp_activity_ratio=...`
(Arch 1 dead-MLP flag if it drops < 0.05) and `query_pairwise_cosine=...`
(Arch 2 collapse flag if it climbs > 0.7).

---

## 4. Log it in your machine-local run log

Per `.claude/rules/distributed-run-log.md`, append a section to **`report/$(whoami)/$(hostname | cut -d. -f1).md`**
(create the file with the standard header if it doesn't exist). **Do not edit any other
machine's file.** Template:

```markdown
---

## {RUN_NAME} diverse-heads FT (ep500 SSL, T=4) | Track A | {YYYY-MM-DD HH:MM}

- **Status:** RUNNING
- **Run:** {RUN_NAME}
- **Experiment:** [`diverse_classifier_heads_post_mae`](../../experiments/diverse_classifier_heads_post_mae.md)
- **Hydra:** `experiment={EXPERIMENT}` {EXTRA, if any, as plain text}
- **Log:** [`{RUN_NAME}_{TAG}.log`](../../logs/track_a/{RUN_NAME}_{TAG}.log)
- **PID:** [`{RUN_NAME}_{TAG}.pid`](../../logs/track_a/{RUN_NAME}_{TAG}.pid)
- **Ckpt:** `checkpoints/track_a/videomaev2+ft/{RUN_NAME}.pt`
- **W&B:** {paste the run URL from the `[wandb] run started:` log line}
- **Metrics:** identity-at-init OK (Arch 3/4); ep1 in progress
```

Flip the status to **DONE** (with best val top1) when it finishes, or **FAILED/STOPPED**
otherwise. Keep it to outcomes — no code or long hyperparameter lists.

---

## 5. Resume after a VM reboot / disconnect

Reuse the **same** `RUN_NAME` and append to the **same** `$LOG`. Resume from the
last checkpoint and let W&B continue the same run:

```bash
WANDB_RESUME=allow PYTHONPATH=src PYTHONUNBUFFERED=1 nohup uv run python -u \
  -m smth2smth.pipelines.train \
  experiment=${EXPERIMENT} \
  model.init_from=${SSL} \
  training.checkpoint_path=${CKPT} \
  training.resume_from=checkpoints/track_a/videomaev2+ft/${RUN_NAME}.last.pt \
  training.wandb.name=${RUN_NAME} training.wandb.project=smth2smth-diverse-heads \
  ${EXTRA} >> "$LOG" 2>&1 &
echo $! > "$PID"
```

---

## 6. Stop / clean up

```bash
kill "$(cat "$PID")" && rm -f "$PID"     # stop a run and remove its pid file
```

When the run ends normally, delete the `.pid` file and set the run-log status to DONE.
The best checkpoint is at `checkpoints/track_a/videomaev2+ft/${RUN_NAME}.pt`; the
rolling last is `…/${RUN_NAME}.last.pt`.
