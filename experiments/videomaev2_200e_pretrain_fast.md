# VideoMAEv2 ViT-B — 200-epoch SSL pretrain (T=16, fast)

> Alternative / replacement spec for `experiments/videomaev2_250e_pretrain.md`.
> The 250-epoch run (`videomaev2_250e_pre_train`, currently at epoch 29/250 on
> espadon, ~21 min/epoch) is **not** stopped by this document; this spec stands
> up a *separate* faster job with non-clashing artifact paths so it can be
> launched whenever a GPU is free (parallel on `raie`, or as the next espadon
> job once the current run finishes / is voluntarily retired).

## 0. Why a new spec (alignment with `new_ideas_tracka.md`)

`new_ideas_tracka.md` (Finding 4, Research Question 3, Recommendations table)
is explicit:

- *"Do not chase more SSL epochs."* Going 200 → 250 buys ≤ ~0.5 pt; even
  200 → 800 caps at ~3 pt on the original SSv2 curve (Tong et al. Fig 5:
  66.4 / 67.9 / 69.6 / 70.3 / 70.6 at 200 / 400 / 800 / 1600 / 2400).
- *"Document the 50/100/150/200/250 SSL-epoch ablation for the prof's report
  (light FT only, no ensembling)."*
- *"More SSL pretraining is low-EV. Allocate the bulk of remaining compute to
  fine-tuning and ensembling."*
- Caveat 4: *"Including val frames in SSL is unambiguously fine. Including the
  test-set frames in SSL pretraining is borderline transductive learning …
  comparisons to literature should note this."*

Compliant pretrain spec ⇒ **cap at 200 epochs** (the last point still on the
not-yet-flat part of the published curve, and the natural stop for the
50/100/150/200 milestone ablation), keep `train+val+test` SSL with the
transductive caveat **explicitly noted in the report**, and recover the
compute lost on a third doubling for fine-tuning / Perceiver-head / ensemble
work (Recommendations table, Days 3–7).

## 1. Differences vs the currently-running 250-ep job

| Knob | Currently running (`videomaev2_250e_pretrain.md`) | This spec |
|------|---------------------------------------------------|-----------|
| Epochs | 250 | **200** (stops at the next clean milestone) |
| Warmup epochs | 15 | **12** (same 6% of schedule) |
| `model.gradient_checkpointing` | `true` | **`false`** |
| `pretrain.batch_size` | 4 | **8** |
| `pretrain.grad_accum_steps` | 16 | **8** (effective batch still **64**) |
| `training.num_workers` | 4 | **8** |
| `persistent_workers` | not wired | **`true`** (needs small code add — §6) |
| `prefetch_factor` | not wired | **`4`** (needs small code add — §6) |
| Encoder ckpt name | `espadon_t16_encoder.pt` | **`espadon_t16_fast_encoder.pt`** |
| Resume state | `…encoder.state.pt` | **`…fast_encoder.state.pt`** |
| Milestone names | `espadon_t16_encoder_ep{N}.pt` | **`espadon_t16_fast_encoder_ep{N}.pt`** |
| W&B run name | `videomaev2_250e_pre_train` | **`videomaev2_200e_pre_train_fast`** |
| Hydra preset | `track_a_ssl_pretrain_e8_t16.yaml` | **`track_a_ssl_pretrain_e9_t16_fast.yaml`** |
| Launch script | `scripts/videomaev2_250e_pretrain.sh` | **`scripts/videomaev2_200e_pretrain_fast.sh`** |

All other knobs (T=16 via 4→16 interpolation, mask ratio 0.75, dual masking
decoder keep 0.50, norm-pix, EMA 0.9999, grad clip 3.0, AMP bf16, AdamW
β=(0.9,0.95), wd=0.05, base LR 1.5e-4 cosine to 0, seed 42) are **unchanged**
— this is a speed + epoch-cap variant, not a recipe change.

Expected wall-clock improvement (from `new_ideas_tracka.md`-style accounting and
the GPU-saturation argument): **~1.6–2× faster per epoch** if grad-ckpt off +
`bs=8` survives the VRAM probe; combined with the 250 → 200 epoch cap, the
total job should finish in **~38–48 % of the 250-ep wall-clock budget**.

## 2. Data

| Item | Setting |
|------|---------|
| Roots | `data/train`, `data/val`, `data/test` (labels ignored) |
| On-disk clips | 4 frames per video (linspace in first 60% of source video) |
| Model input | **T=16** — `source_num_frames=4`, `temporal_expand_mode=interpolation` |
| `include_val_in_pretrain` | `true` (val frames OK per `new_ideas_tracka.md` caveat 4) |
| Augmentation (training) | `random_crop` + `crop_padding=32` only; **no** horizontal flip |
| Transductive note | Test frames are included in SSL, as in the 250-ep run; **must be flagged in the report** when comparing to published VideoMAE numbers. |

## 3. Model & loss

Identical to the 250-ep spec:

| Item | Value |
|------|-------|
| Architecture | `video_mae_vit`, variant `vit_b` |
| `tube_t` | 1 |
| `patch_size` | 16 |
| Masking | V2 dual masking: encoder tube ratio **0.75**, decoder cell keep **0.50** (2×2 cells) |
| Target | Per-cube normalized pixel MSE (`norm_pix=true`) |
| `gradient_checkpointing` | **`false`** ← only change vs 250-ep |

## 4. Optimisation

| Item | Value |
|------|-------|
| Optimiser | AdamW, β=(0.9, 0.95), `weight_decay=0.05` |
| Base LR | `1.5e-4` |
| Schedule | Linear warmup **12** epochs → cosine decay to `min_lr=0` |
| Per-GPU batch | **8** |
| Grad accum | **8** → effective batch **64** (same as 250-ep) |
| AMP | bfloat16 |
| Grad clip | global L2 norm **3.0** |
| EMA | enabled, decay **0.9999** (encoder checkpoints use EMA weights) |
| Epochs | **200** |
| Seed | 42 |

## 5. Checkpoints & monitoring

Distinct paths so this job **never overwrites** the running 250-ep artifacts.

| Artifact | Path |
|----------|------|
| Latest encoder (EMA) | `checkpoints/track_a/ssl/espadon_t16_fast_encoder.pt` |
| Resume state | `checkpoints/track_a/ssl/espadon_t16_fast_encoder.state.pt` |
| Milestones (50, 100, 150, 200) | `espadon_t16_fast_encoder_ep{N}.pt` |

The four milestone checkpoints (50/100/150/200) are the artifacts the prof's
**SSL-epoch ablation** runs over (one light FT each — `new_ideas_tracka.md`
Recommendations table, Day 6–7). The 250-checkpoint slot is intentionally
dropped: the doc's headline finding is that 200 → 250 is < 0.5 pt and not
worth running.

**Weights & Biases**

- Project: `smth2smth`
- Run name: **`videomaev2_200e_pre_train_fast`**
- Group: `track_a_videomaev2_pretrain` (same as 250-ep so both runs are
  graphed against each other in W&B)
- API key: repo-root `.env` (`WANDB_API_KEY`, gitignored)
- Logged scalars: `train/loss`, `train/lr`, `train/epoch_avg_loss`
- W&B config: only **active** augmentation keys (no disabled flags) — and an
  extra tag `"fast-restart"` with the diffed knobs (`grad_ckpt=false`,
  `bs=8/ga=8`, `num_workers=8`, `persistent_workers=true`, `prefetch=4`,
  `epochs=200`) so post-hoc comparisons to the 250-ep run are unambiguous.

## 6. Required code change (one file)

`src/smth2smth/pipelines/pretrain_videomae.py` currently builds the
`DataLoader` at line ~185 with no `persistent_workers` or `prefetch_factor`.
Both knobs need to be plumbed through `cfg.training` (or `cfg.pretrain`) and
forwarded only when `num_workers > 0`:

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

This is **purely additive** — the 250-ep run is unaffected because it never
sets the new keys, so it keeps the PyTorch defaults
(`persistent_workers=False`, `prefetch_factor=2`).

## 7. Hydra preset (new file)

`configs/experiment/track_a_ssl_pretrain_e9_t16_fast.yaml` — copy of
`track_a_ssl_pretrain_e8_t16.yaml` with the following overrides applied:

```yaml
# @package _global_
# VideoMAEv2 ViT-B SSL pretrain 200 ep @ T=16, fast variant.
# Spec: experiments/videomaev2_200e_pretrain_fast.md
defaults:
  - override /model: video_mae_vit
  - override /pretrain: videomae
  - override /augment: videomae_pretrain_t16

model:
  variant: vit_b
  tube_t: 1
  patch_size: 16
  mlp_ratio: 4.0
  gradient_checkpointing: false        # ← off (was true)

seed: 42

dataset:
  num_frames: 16

pretrain:
  epochs: 200                          # ← 200 (was 250)
  warmup_epochs: 12                    # ← 12 (was 15)
  batch_size: 8                        # ← 8 (was 4)
  grad_accum_steps: 8                  # ← 8 (was 16); effective batch 64
  num_frames: 16
  source_num_frames: 4
  temporal_expand_mode: interpolation
  lr: 0.00015
  weight_decay: 0.05
  mask_ratio: 0.75
  norm_pix: true
  drop_path_rate: 0.0
  dual_masking: true
  decoder_keep_ratio: 0.50
  decoder_cell_h: 2
  decoder_cell_w: 2
  include_val_in_pretrain: true
  max_grad_norm: 3.0
  ema_enabled: true
  ema_decay: 0.9999
  checkpoint_milestones: [50, 100, 150, 200]   # ← drop 250
  checkpoint_path: ${hydra:runtime.cwd}/checkpoints/track_a/ssl/espadon_t16_fast_encoder.pt
  auto_resume: true
  wandb_enabled: true
  wandb_project: smth2smth
  wandb_run_name: videomaev2_200e_pre_train_fast
  wandb_group: track_a_videomaev2_pretrain

training:
  num_workers: 8                       # ← 8 (was 4)
  persistent_workers: true             # ← new
  prefetch_factor: 4                   # ← new
  device: cuda
  amp: true
  batch_size: 8
```

## 8. Launch sequence (with mandatory VRAM probe)

The 250-ep run's launch banner reported **~23.15 / 23.54 GiB free** at
`bs=4` + grad-ckpt on. Turning grad-ckpt off **and** doubling micro-batch is
non-trivially VRAM-heavier; an OOM probe is mandatory.

```bash
cd /Data/romain.poggi/smth2smth
set -a && source .env && set +a

# Step 1 — short VRAM probe (~2-3 min, no W&B, 32 clips, 1 epoch).
PYTHONPATH=src .venv/bin/python -u -m smth2smth.pipelines.pretrain_videomae \
  experiment=track_a_ssl_pretrain_e9_t16_fast track=a \
  pretrain.max_videos=32 pretrain.epochs=1 pretrain.wandb_enabled=false
```

If the probe OOMs, fall back to the **conservative variant**: keep
`pretrain.batch_size=4` / `grad_accum_steps=16`, but still drop
`model.gradient_checkpointing=false` — this alone is typically **+25–40 %**
throughput at no VRAM risk above the current 250-ep job.

```bash
# Step 2 — real launch (only if Step 1 finished without OOM).
export BATCH_TAG="$(date +%Y%m%d)"
nohup bash scripts/videomaev2_200e_pretrain_fast.sh \
  >> "logs/pre_train/videomaev2_200e_pretrain_fast_${BATCH_TAG}.log" 2>&1 &
echo $! > "logs/pre_train/videomaev2_200e_pretrain_fast_${BATCH_TAG}.pid"
```

`scripts/videomaev2_200e_pretrain_fast.sh` is a copy of the existing
`scripts/videomaev2_250e_pretrain.sh` with the Hydra preset and log/pid base
names swapped to `videomaev2_200e_pretrain_fast`.

Monitor:

```bash
tail -f logs/pre_train/videomaev2_200e_pretrain_fast_${BATCH_TAG}.log
kill -0 $(cat logs/pre_train/videomaev2_200e_pretrain_fast_${BATCH_TAG}.pid) && echo RUNNING
```

## 9. Out of scope / explicitly deferred

- **No recipe change.** Mask ratio, LR, optimiser, EMA, grad clip, AMP dtype,
  dataset roots, augmentation list, T=16 interpolation, seed — all identical
  to the 250-ep spec. Per `new_ideas_tracka.md` Recommendations Day 1 the
  recipe changes belong to **fine-tuning**, not pretrain.
- **No alternative SSL objective.** MAM², MotionMAE, MGMAE, MVD, MOFO are
  catalogued in `new_ideas_tracka.md` §RQ3(e) but flagged "non-trivial to add
  in 2 days" and gated behind teacher / optical-flow infrastructure we do not
  currently have. Out of scope for this spec.
- **No 250-th milestone.** The 250-ep checkpoint is intentionally dropped:
  the doc's central pretrain claim is 200 → 250 is < 0.5 pt and not worth
  running.
- **No T=4 fallback.** The abandoned `track_a_ssl_pretrain_e8` T=4 line is
  not revived; T=16 via interpolation stays canonical (as in the 250-ep spec).
- **Excluding test frames from SSL.** Allowed by the rules and noted as
  *"borderline transductive"* in caveat 4 — kept in this spec for parity with
  the 250-ep run, but the report must explicitly disclose that test frames
  were seen by SSL when comparing to published VideoMAE Top-1 numbers.

## 10. Success criteria

- Loss curve in W&B decreases smoothly with no recurring NaN/OOM
- All four milestone checkpoints exist at epochs 50, 100, 150, 200
- Final `espadon_t16_fast_encoder.pt` loads into `video_mae_vit` with
  `num_frames=16`, `tube_t=1`
- Per-epoch wall-clock < **~14 min** (≥ 1.5× faster than the 250-ep job's
  ~21 min/epoch) — if not met, document the actual speedup and decide
  whether the fast variant is worth the parallel slot
- The 50/100/150/200 milestones are usable as inputs to the prof's SSL-epoch
  FT ablation (`new_ideas_tracka.md` Recommendations Day 6–7)
