# VideoMAE v2 ViT-B — 300-epoch SSL pretrain, **native T=4**, train+val+test

> **Role in the frame-expansion ablation.** This run is the *native-T=4* leg
> of a 2-arm ablation against the currently-running 250-epoch interpolated-T=16
> job. The third encoder, `e_r1_encoder.pt` (≡ `sole_encoder.pt` on Piranha,
> Sole rescue 2026-05-16), exists already and is the production baseline for
> the HC arms; it is **not re-run** here, only referenced.
>
> Replication-mode expansion (`temporal_expand_mode=replication`) is
> **explicitly out of scope**: with `tube_t=1`, replicated tubes are
> byte-equal to their source frame, so any unmasked tube in a replication
> group trivially reveals the masked tubes from the same group — an
> uninformative SSL objective. The honest "native temporal granularity" leg
> is T=4 with no expansion.

## 0. The ablation question this run answers

**Does interpolating T=4 → T=16 at SSL time (currently-running 250e job)
produce a downstream-FT encoder meaningfully different from / better than
the same v2 recipe at native T=4?**

Holding everything else constant (ViT-B, v2 dual masking, mask 0.75 static,
EMA 0.9997, effective batch 64, LR 1.5e-4, 15-epoch warmup, AdamW
β=(0.9, 0.95), weight decay 0.05, drop-path 0.0, max-grad-norm 3.0,
`random_crop`+padding-32 aug, no horizontal flip, train+val+test SSL roots),
the **only variable that changes** between this run and the currently-running
250e job is the frame budget:

- Currently-running 250e: `num_frames=16`, `source_num_frames=4`,
  `temporal_expand_mode=interpolation` (12 of 16 frames are RGB blends).
- **This run: `num_frames=4`, no expansion (no `source_num_frames`, no
  `temporal_expand_mode`).**

This run is **also intentionally different from `e_r1`** on four orthogonal
axes — data (val included), mask ratio (0.75 vs 0.90), EMA (on vs off),
optim recipe (eff_bs 64 / lr 1.5e-4 vs 128 / 7.5e-5) — so its encoder is
expected to land in a different basin than `e_r1`'s and contribute genuine
diversity to a downstream ensemble (per `new_ideas_tracka.md`, Day 3-5: the
ensemble should mix recipes, not just seeds).

## 1. Data

| Item | Setting |
|------|---------|
| SSL roots | `data/train`, `data/val`, `data/test` (labels ignored) — set by `include_val_in_pretrain=true` |
| On-disk clips | 4 frames per video (linspace in first 60% of source video, as produced by the pipeline) |
| Model input | **T=4 (native)** — no `source_num_frames`, no `temporal_expand_mode`. |
| Augmentation | `random_crop=true`, `crop_padding=32`, **no** horizontal flip (SSv2 is direction-sensitive) |
| Transductive note | Test frames are included in SSL (same as current 250e). Per `new_ideas_tracka.md` caveat 4 this is borderline-transductive; report must disclose it when comparing to published VideoMAE numbers. |

The dataset cardinality is **identical** to `e_r1` (which also uses 4-frame
native clips) plus the val frames — i.e. roughly `51,906 + |val|` clips.
Confirm at launch with `[videomae] using <N> unlabeled clips (T=4).`

## 2. Architecture

Identical to `e_r1` and current 250e — only the input frame count changes:

| Item | Value |
|------|-------|
| Family | `video_mae_vit`, variant **`vit_b`** (12 layers, 768 dim, 12 heads, ~87 M params) |
| `tube_t` | **1** (matches e_r1 and current 250e) |
| `patch_size` | 16 |
| `mlp_ratio` | 4.0 |
| `drop_path_rate` | 0.0 (pretrain standard) |
| `dropout` | 0.0 |
| `head` | `mean` |
| `gradient_checkpointing` | **`false`** (speed; see §5 VRAM probe) |
| `freeze_backbone` | `false` |
| Tokens per clip | 4 × 14 × 14 = **784** (mask 0.75 → ~196 visible) |

## 3. SSL objective (VideoMAE v2)

| Item | Value |
|------|-------|
| Mask type | tube mask (random) |
| `mask_ratio` | **0.75 static** (no schedule) |
| `dual_masking` | **true** (v2) — decoder running-cell mask keeps 50 % of N at 2×2 cells |
| `decoder_keep_ratio` | 0.50 |
| `decoder_cell_h`, `decoder_cell_w` | 2, 2 |
| `norm_pix` | true (per-cube pixel normalisation) |
| Loss | per-cube normalised pixel MSE, **invisible-only** (v2 Table 1 convention) |

## 4. Optimisation

| Item | Value |
|------|-------|
| Optimiser | **AdamW**, β=(0.9, 0.95), `weight_decay=0.05` |
| Base LR | **1.5e-4** (V2 paper value at effective batch 64) |
| LR schedule | Linear warmup **15 epochs** → cosine decay to `min_lr=0` |
| `pretrain.batch_size` (per GPU) | **8** |
| `pretrain.grad_accum_steps` | **8** → effective batch **64** (same as current 250e) |
| AMP | bfloat16 |
| `max_grad_norm` | 3.0 (global L2) |
| `ema_enabled` | **true**, `ema_decay=0.9997` (encoder checkpoints export EMA weights) — calibrated for a ~244 k-opt-step / 300-epoch run: half-life ~2.85 epochs, 90 %-mass window ~9.5 epochs, init-contamination clears by ~epoch 19. Keeps roughly the same "fraction of training in the EMA window" as 0.9995 would have at 200 ep. The original `e_r1`-era / diffusion-borrowed value of 0.9999 leaks the random init into the EMA past epoch 50 even at 300 ep; do **not** raise this back. |
| `drop_path_rate` | 0.0 |
| Epochs | **300** (one epoch past the `new_ideas_tracka.md` plateau guidance; +100 ep over `e_r1` to capture the last bit of the curve and to give the EMA a wider stable window before the final milestone) |
| Seed | 42 |

### Data loader (requires small one-time code change)

`src/smth2smth/pipelines/pretrain_videomae.py` currently builds the
`DataLoader` (~line 185) without exposing `persistent_workers` or
`prefetch_factor`. This spec requires both, plumbed through `cfg.training`:

```python
loader_kwargs = dict(
    batch_size=int(pcfg.batch_size),
    shuffle=True,
    num_workers=int(cfg.training.num_workers),
    pin_memory=(device.type == "cuda"),
    drop_last=True,
)
if int(cfg.training.num_workers) > 0:
    loader_kwargs["persistent_workers"] = bool(
        cfg.training.get("persistent_workers", False)
    )
    loader_kwargs["prefetch_factor"] = int(
        cfg.training.get("prefetch_factor", 2)
    )
loader = DataLoader(dataset, **loader_kwargs)
```

This is **purely additive** — existing runs (e_r1 was already finished;
current 250e never sets the new keys) keep PyTorch defaults
(`persistent_workers=False`, `prefetch_factor=2`).

Loader settings for this run:

| Item | Value |
|------|-------|
| `training.num_workers` | **8** |
| `training.persistent_workers` | **true** (new) |
| `training.prefetch_factor` | **4** (new) |
| `training.device` | `cuda` |
| `training.amp` | true |

## 5. Hardware / VRAM probe (mandatory before the long launch)

At native T=4 + `tube_t=1` + `gradient_checkpointing=false` + per-GPU `bs=8`,
this run should fit comfortably in a 24 GiB 3090 (the 250e job uses **bs=4**
at T=16 with grad-ckpt on; halving the frame count and quadrupling the
visible-token budget at `bs=8` still ends up token-wise lighter than the
250e job). Confirm with a 1-epoch / 32-clip probe before committing.

```bash
cd /Data/romain.poggi/smth2smth
set -a && source .env && set +a
PYTHONPATH=src .venv/bin/python -u -m smth2smth.pipelines.pretrain_videomae \
  experiment=track_a_ssl_pretrain_e9_t4_native track=a \
  pretrain.max_videos=32 pretrain.epochs=1 pretrain.wandb_enabled=false
```

If the probe OOMs (unlikely): drop `pretrain.batch_size=4
pretrain.grad_accum_steps=16` (still effective batch 64). If it still OOMs:
re-enable `model.gradient_checkpointing=true` and accept the throughput hit.

## 6. Checkpoints & artifact paths (non-clashing with all other runs)

| Artifact | Path |
|----------|------|
| Latest encoder (EMA) | `checkpoints/track_a/ssl/videomaev2_t4native_encoder.pt` |
| Resume state | `checkpoints/track_a/ssl/videomaev2_t4native_encoder.state.pt` |
| Milestone checkpoints (epochs 50, 100, 150, 200, 250, 300) | `checkpoints/track_a/ssl/videomaev2_t4native_encoder_ep{N}.pt` |
| Pretrain log | `logs/pre_train/videomaev2_t4native_pretrain_${BATCH_TAG}.log` |
| Pretrain PID | `logs/pre_train/videomaev2_t4native_pretrain_${BATCH_TAG}.pid` |

**Names are deliberately distinct from**:
- `e_r1_encoder.pt` / `sole_encoder.pt` (the existing baseline encoder used
  in `instruction_hc_ablation.md`),
- `espadon_t16_encoder.pt` (the currently-running 250e job),
- `espadon_encoder.pt` (the abandoned E8 T=4 line).

No symlinks. No overwrites.

## 7. Weights & Biases

W&B is **mandatory** for this run. Tracker setup steps before the launch:

1. **Confirm key.** `cat $REPO_ROOT/.env | grep WANDB_API_KEY` returns a
   non-empty line. If absent, paste your key from
   <https://wandb.ai/authorize> into `.env` (gitignored). The launch script
   `set -a; source .env; set +a`s the file.
2. **Confirm the venv has wandb.** `.venv/bin/python -c "import wandb;
   print(wandb.__version__)"` should print without error. If missing, run
   `uv sync` (wandb is already pinned in `pyproject.toml` / `uv.lock`).
3. **Confirm CLI auth (optional but recommended).**
   `.venv/bin/wandb login` — paste the same key; this writes
   `~/.netrc` so the W&B SDK can authenticate independently of the env var
   should the env be stripped.
4. **Project — separate from the main dashboard.** Set
   `pretrain.wandb_project=smth2smth-frame-ablation` (created automatically
   on first `wandb.init`). This keeps the 3-arm frame-expansion ablation
   visually grouped and prevents the main `smth2smth` board from getting
   noisy.
5. **First-run check.** After `[videomae]` logs the first epoch step, open
   `https://wandb.ai/<entity>/smth2smth-frame-ablation/runs/` and confirm
   the run named `videomaev2_t4_native_300e_pretrain` is there. If the run
   does not appear within ~2 min of launch but the local log is advancing,
   the W&B init failed silently — `grep -i 'wandb' <log>` for the cause
   (usually a stale `~/.netrc` from a different account).

W&B run metadata:

| Item | Value |
|------|-------|
| `pretrain.wandb_enabled` | **true** |
| `pretrain.wandb_project` | **`smth2smth-frame-ablation`** |
| `pretrain.wandb_run_name` | **`videomaev2_t4_native_300e_pretrain`** |
| `pretrain.wandb_group` | **`frame_expansion_ablation`** (shared with the 250e job once it is also tagged into this group) |
| Logged scalars | `train/loss`, `train/lr`, `train/epoch_avg_loss` (existing pipeline-side) |
| Extra W&B tags | `["t4-native", "vit-b", "v2-dual-masking", "train+val+test", "ablation-arm-native"]` |

To keep the comparison interpretable, **the currently-running 250e job
should also be tagged into `wandb_group=frame_expansion_ablation`** (a one-line
W&B-UI re-tag after the fact is fine; do not stop the run). If a third
ablation point is desired later (e.g. T=8 interpolation, half-budget), it
joins the same group.

## 8. Hydra preset (new file)

`configs/experiment/track_a_ssl_pretrain_e9_t4_native.yaml`:

```yaml
# @package _global_
# E9 — VideoMAE v2 ViT-B SSL pretrain, native T=4, train+val+test.
# Spec: experiments/videomaev2_t4_native_300e_pretrain.md
# Role: native-T=4 arm of the frame-expansion ablation (vs current 250e
# interpolated-T=16 arm and the existing e_r1_encoder.pt baseline).
defaults:
  - override /model: video_mae_vit
  - override /pretrain: videomae
  - override /augment: videomae_pretrain_t16   # same aug list; T independent

model:
  variant: vit_b
  tube_t: 1
  patch_size: 16
  mlp_ratio: 4.0
  gradient_checkpointing: false        # ← speed (vs e_r1's true)

augment:
  random_crop: true
  crop_padding: 32
  random_horizontal_flip: false

seed: 42

dataset:
  num_frames: 4                        # ← native T=4 (vs current 250e's 16)

pretrain:
  epochs: 300                          # ← +100 over e_r1; see §4 "Epochs" row
  warmup_epochs: 15
  batch_size: 8                        # eff bs 64 with ga 8
  grad_accum_steps: 8
  num_frames: 4
  # NO source_num_frames, NO temporal_expand_mode — native
  lr: 1.5e-4
  weight_decay: 0.05
  mask_ratio: 0.75
  norm_pix: true
  drop_path_rate: 0.0

  dual_masking: true
  decoder_keep_ratio: 0.50
  decoder_cell_h: 2
  decoder_cell_w: 2

  include_val_in_pretrain: true        # ← train+val+test (vs e_r1's train+test)
  max_grad_norm: 3.0

  ema_enabled: true                    # ← EMA on (vs e_r1's off)
  ema_decay: 0.9997                    # calibrated for ~244k opt steps / 300 ep (see §4)

  checkpoint_milestones: [50, 100, 150, 200, 250, 300]
  checkpoint_path: ${hydra:runtime.cwd}/checkpoints/track_a/ssl/videomaev2_t4native_encoder.pt
  auto_resume: true

  wandb_enabled: true
  wandb_project: smth2smth-frame-ablation
  wandb_run_name: videomaev2_t4_native_300e_pretrain
  wandb_group: frame_expansion_ablation

training:
  num_workers: 8
  persistent_workers: true             # ← requires §4 code change
  prefetch_factor: 4                   # ← requires §4 code change
  device: cuda
  amp: true
  batch_size: 8
```

## 9. Launch script (new file)

`scripts/videomaev2_t4_native_pretrain.sh`:

```bash
#!/usr/bin/env bash
# VideoMAE v2 ViT-B SSL pretrain — native T=4, 300 ep, train+val+test.
# Spec: experiments/videomaev2_t4_native_300e_pretrain.md
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1 PYTHONPATH=src

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a; source "${REPO_ROOT}/.env"; set +a
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "Missing WANDB_API_KEY — add it to ${REPO_ROOT}/.env" >&2
  exit 1
fi

PY="${REPO_ROOT}/.venv/bin/python"
[[ -x "$PY" ]] || { echo "missing .venv — run: uv sync" >&2; exit 1; }

BATCH_TAG="${BATCH_TAG:-$(date +%Y%m%d)}"
LOG_DIR="${REPO_ROOT}/logs/pre_train"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/videomaev2_t4native_pretrain_${BATCH_TAG}.log"

echo "[videomaev2_t4native] $(date -Is) pretrain ViT-B 300 ep T=4 native (train+val+test)"
"$PY" -u -m smth2smth.pipelines.pretrain_videomae \
  experiment=track_a_ssl_pretrain_e9_t4_native track=a

test -s checkpoints/track_a/ssl/videomaev2_t4native_encoder.pt
echo "[videomaev2_t4native] $(date -Is) Done."
```

Make executable: `chmod +x scripts/videomaev2_t4_native_pretrain.sh`.

## 10. Launch + monitor

```bash
cd /Data/romain.poggi/smth2smth
export BATCH_TAG="$(date +%Y%m%d)"

# Step 1 — VRAM probe (§5). Skip only if you already ran it for this exact venv/GPU combo.
PYTHONPATH=src .venv/bin/python -u -m smth2smth.pipelines.pretrain_videomae \
  experiment=track_a_ssl_pretrain_e9_t4_native track=a \
  pretrain.max_videos=32 pretrain.epochs=1 pretrain.wandb_enabled=false

# Step 2 — real launch (only if Step 1 finished without OOM).
nohup bash scripts/videomaev2_t4_native_pretrain.sh \
  >> "logs/pre_train/videomaev2_t4native_pretrain_${BATCH_TAG}.log" 2>&1 &
echo $! > "logs/pre_train/videomaev2_t4native_pretrain_${BATCH_TAG}.pid"

# Step 3 — monitor.
tail -f logs/pre_train/videomaev2_t4native_pretrain_${BATCH_TAG}.log
kill -0 $(cat logs/pre_train/videomaev2_t4native_pretrain_${BATCH_TAG}.pid) && echo RUNNING
```

The script auto-resumes from `…_t4native_encoder.state.pt` end-of-epoch
state if killed and re-launched (`auto_resume=true`).

## 11. Success criteria

- W&B run `videomaev2_t4_native_300e_pretrain` appears in
  `https://wandb.ai/<entity>/smth2smth-frame-ablation/` within 2 min of
  launch, and `train/epoch_avg_loss` is logged at the end of each epoch.
- Loss curve decreases smoothly without recurring NaN / OOM.
- All six milestone checkpoints exist:
  `videomaev2_t4native_encoder_ep{50,100,150,200,250,300}.pt`.
- Final `videomaev2_t4native_encoder.pt` loads into `video_mae_vit` with
  `num_frames=4`, `tube_t=1`, `variant=vit_b` (sanity: token count = 784,
  visible after mask = ~196).
- Per-epoch wall-clock < **~10 min** at native T=4 + grad-ckpt off + bs=8
  (rough budget: ~3× lighter per step than the 250e job at T=16 + grad-ckpt
  on; total run for 300 ep < ~50 h).

## 12. Out of scope

- **Replication-mode expansion** (`temporal_expand_mode=replication`):
  trivial reconstruction with `tube_t=1`, excluded by design (see header
  note).
- **Higher `tube_t`** (e.g. 2 or 4) at native T=4: reduces temporal token
  count below the 4-position resolution needed for SSv2 direction-sensitive
  classes. Hold for a separate experiment.
- **Mask-ratio schedules** (e.g. 0.95 → 0.75 cosine): orthogonal axis;
  re-running with `mask_ratio_schedule=true` would conflate the frame-budget
  ablation with the curriculum question. Hold for a separate experiment.
- **Alternative SSL objectives** (MotionMAE, MAM², MGMAE): per
  `new_ideas_tracka.md` §RQ3(e), non-trivial to add in < 2 days and gated
  behind optical-flow / teacher infrastructure we don't have. Hold.
- **Resume across the e_r1 / 250e / t4-native arms**: each arm has a
  distinct `checkpoint_path` and `state.pt`; do not point this run at the
  others' state files even by accident.

## 13. Downstream usage (informational)

Once this encoder lands, the natural FT consumers per `new_ideas_tracka.md`
Recommendations Day 1–5:

| Ensemble member | Encoder | Head | T at FT |
|-----------------|---------|------|---------|
| A | `e_r1_encoder.pt` (existing) | attentive probe | 4 |
| B | `videomaev2_t4native_encoder.pt` (this run) | Perceiver / cross-attention pooling | 4 |
| C | `espadon_t16_encoder.pt` (current 250e, when it lands) | attentive probe | 16 (interp) |
| D (optional) | `videomaev2_t4native_encoder.pt` (this run) | attentive probe, different aug seed | 4 |

This pairing maximises encoder × head × frame-budget diversity for the
learned-weight logit ensemble in `new_ideas_tracka.md` Recommendations Day
5–6. No FT yaml is committed by *this* spec; FT configs are a separate
experiment.
