# VideoMAEv2 ViT-B — 250-epoch SSL pretrain (T=16)

## 1. Purpose

Train a **VideoMAE v2** ViT-B encoder from scratch on the Track-A unlabeled frame corpus
(`train + val + test`), with **16 input frames** built from the professor’s **4-frame**
clips via **temporal interpolation**. This run supersedes the abandoned **E8 / espadon**
T=4 pretrain (`track_a_ssl_pretrain_e8`); that job is not used for checkpoints or analysis.

Downstream: supervised champion fine-tune on honest official val (FT preset TBD).

## 2. Data

| Item | Setting |
|------|---------|
| Roots | `data/train`, `data/val`, `data/test` (labels ignored) |
| On-disk clips | 4 frames per video (linspace in first 60% of source video) |
| Model input | **T=16** — `source_num_frames=4`, `temporal_expand_mode=interpolation` |
| Augmentation (training) | `random_crop` + `crop_padding=32` only; **no** horizontal flip |

Interpolation places 16 samples along the 4-frame timeline and linearly blends adjacent
RGB frames (PIL `Image.blend`). Alternative `replication` (4× repeat per frame) is
available in config but **not** used here.

## 3. Model & loss

| Item | Value |
|------|-------|
| Architecture | `video_mae_vit`, variant `vit_b` |
| `tube_t` | 1 |
| `patch_size` | 16 |
| Masking | V2 dual masking: encoder tube ratio **0.75**, decoder cell keep **0.50** (2×2 cells) |
| Target | Per-cube normalized pixel MSE (`norm_pix=true`) |
| `gradient_checkpointing` | true (encoder) |

## 4. Optimisation

| Item | Value |
|------|-------|
| Optimiser | AdamW, β=(0.9, 0.95), `weight_decay=0.05` |
| Base LR | `1.5e-4` |
| Schedule | Linear warmup **15** epochs → cosine decay to `min_lr=0` |
| Batch | **4** per GPU × **grad_accum 16** → effective batch **64** |
| AMP | bfloat16 |
| Grad clip | global L2 norm **3.0** |
| EMA | enabled, decay **0.9999** (encoder checkpoints use EMA weights) |
| Epochs | **250** |
| Seed | 42 |

## 5. Checkpoints & monitoring

| Artifact | Path |
|----------|------|
| Latest encoder (EMA) | `checkpoints/track_a/ssl/espadon_t16_encoder.pt` |
| Resume state | `checkpoints/track_a/ssl/espadon_t16_encoder.state.pt` |
| Milestones (epochs 50, 100, 150, 200, 250) | `espadon_t16_encoder_ep{N}.pt` |

**Weights & Biases**

- Project: `smth2smth`
- Run name: `videomaev2_250e_pre_train`
- API key: repo-root `.env` (`WANDB_API_KEY`, gitignored)
- Logged scalars: `train/loss`, `train/lr`, `train/epoch_avg_loss`
- W&B config: only **active** augmentation keys (no disabled flags)

## 6. Hydra preset & launch

**Preset:** `configs/experiment/track_a_ssl_pretrain_e8_t16.yaml`  
(Hydra name retained for compatibility; see this doc for the canonical spec.)

```bash
cd /path/to/smth2smth
export BATCH_TAG="$(date +%Y%m%d)"

# Load WANDB_API_KEY from .env
set -a && source .env && set +a

nohup bash scripts/videomaev2_250e_pretrain.sh \
  >> "logs/pre_train/videomaev2_250e_pretrain_${BATCH_TAG}.log" 2>&1 &
echo $! > "logs/pre_train/videomaev2_250e_pretrain_${BATCH_TAG}.pid"
```

The launch script also writes to `logs/pre_train/` by default (creates the directory if missing).

Monitor:

```bash
tail -f logs/pre_train/videomaev2_250e_pretrain_${BATCH_TAG}.log
kill -0 $(cat logs/pre_train/videomaev2_250e_pretrain_${BATCH_TAG}.pid) && echo RUNNING
```

## 7. Out of scope / deferred

- E8 T=4 pretrain logs and `espadon_encoder.pt` — **ignored**
- Champion FT at T=16 — separate experiment file when pretrain completes
- `replication` temporal expand — not used (interpolation only)

## 8. Success criteria (pretrain)

- Loss curve decreases smoothly in W&B without recurrent NaN/OOM
- Milestone checkpoints exist at epochs 50, 100, 150, 200, 250
- Final `espadon_t16_encoder.pt` loads into `video_mae_vit` with `num_frames=16`, `tube_t=1`
