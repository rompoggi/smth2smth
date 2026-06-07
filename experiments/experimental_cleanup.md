# Experimental cleanup — seed-43 MAE scaling FT fleet (May 2026)

Controlled replication of the official SSv2 VideoMAE fine-tune recipe (`track_a_videomae_official_ssv2_ft`, `T=4`, `tube_t=1`) with **seed 43** instead of 42. Reference run: [`mae50-ft-f4`](../../logs/track_a/mae50-ft-f4_20260524.log) (mean-pool, SSL ep50, seed 42, best val top1 **0.4519**).

**Goal:** (1) Re-establish the pretrain-epoch scaling curve under a new seed; (2) Pair Perceiver Q16 heads on the same seed at selected SSL depths for a fair head-vs-backbone comparison.

**Coordinator:** gymnote · **Launcher:** [`scripts/launch_ft_mae_scaling_s43_fleet.sh`](../scripts/launch_ft_mae_scaling_s43_fleet.sh) · **Host map:** [`report/romain.poggi/gymnote.md`](../report/romain.poggi/gymnote.md) (fleet section).

## W&B

| Field | Value |
|-------|--------|
| Project | `smth2smth-frame-ablation` |
| Group | `ft-mae-scaling` |
| Config keys | `head_type` (`meanpool` \| `perceiverQ16`), `pretrain_epochs` (50–500) |

## Recipe (unchanged vs mae50-ft-f4)

- Hydra: `experiment=track_a_videomae_official_ssv2_ft` (mean-pool) or `track_a_diverse_arch2_perceiver` (Perceiver Q16)
- `seed=43`, `dataset.num_frames=4`, `model.tube_t=1`

### Validation protocol (no train-on-val)

| Flag | Value | Effect |
|------|--------|--------|
| `dataset.use_official_val` | `true` | Validate on full `data/val` (6745 clips), not an 80/20 split of train |
| `dataset.include_val_in_train` | `false` | **No** val clips in the training set |
| `dataset.official_val_holdout_ratio` | `0.0` | **No** 90/10 val carve-out for train (unlike `*_stab` / holdout presets at `0.1`) |

Expected log line at startup (full run): `train=44993 (train_dir=44993), val=6745` — if you see `+ val_dir=` in the train count, stop the job.

Launchers also pass `dataset.include_val_in_train=false` and `dataset.official_val_holdout_ratio=0` on the CLI for defense in depth.
- SSL init: `checkpoints/track_a/ssl/pretrain/videomaev2_t4native_encoder_ep{N}.pt`
- Optim / aug: `configs/train/videomae_official_ssv2.yaml` + `configs/augment/official_videomae_ssv2.yaml`
- FT output: `checkpoints/track_a/videomaev2+ft/<run>.pt`
- Logs: `logs/track_a/<run>_YYYYMMDD.log`

## Fleet manifest (seed 43 — complete)

See [`logs/track_a/ft_mae_scaling_s43/manifest.txt`](../logs/track_a/ft_mae_scaling_s43/manifest.txt) for per-host best val. Checkpoints: `checkpoints/track_a/ft_mae_scaling_s43/*.pt`.

## Fleet manifest (seed 44 — 14 GPUs, skip mae050)

**Launcher:** [`scripts/launch_ft_mae_scaling_s44_fleet.sh`](../scripts/launch_ft_mae_scaling_s44_fleet.sh)

| Host | Run | SSL ep |
|------|-----|--------|
| gardon | `meanpool-mae100-s44` | 100 |
| gymnote | `meanpool-mae150-s44` | 150 |
| labre | `meanpool-mae200-s44` | 200 |
| lieu | `meanpool-mae250-s44` | 250 |
| lotte | `meanpool-mae300-s44` | 300 |
| mulet | `meanpool-mae350-s44` | 350 |
| murene | `meanpool-mae400-s44` | 400 |
| piranha | `meanpool-mae450-s44` | 450 |
| raie | `meanpool-mae500-s44` | 500 |
| requin | `perceiverQ16-mae100-s44` | 100 |
| rouget | `perceiverQ16-mae200-s44` | 200 |
| sole | `perceiverQ16-mae300-s44` | 300 |
| thon | `perceiverQ16-mae400-s44` | 400 |
| truite | `perceiverQ16-mae500-s44` | 500 |

Skipped: `meanpool-mae050-s44` (no 15th GPU; use s43 mae050 on gardon for that point).

## Fleet manifest (15 GPUs)

### Set 1 — Mean-pool scaling (seed 43)

| Host | Run | SSL ep | W&B name |
|------|-----|--------|----------|
| gardon | `meanpool-mae050-s43` | 50 | `meanpool-mae050-s43` |
| gymnote | `meanpool-mae100-s43` | 100 | `meanpool-mae100-s43` |
| labre | `meanpool-mae150-s43` | 150 | `meanpool-mae150-s43` |
| lieu | `meanpool-mae200-s43` | 200 | `meanpool-mae200-s43` |
| lotte | `meanpool-mae250-s43` | 250 | `meanpool-mae250-s43` |
| mulet | `meanpool-mae300-s43` | 300 | `meanpool-mae300-s43` |
| murene | `meanpool-mae350-s43` | 350 | `meanpool-mae350-s43` |
| piranha | `meanpool-mae400-s43` | 400 | `meanpool-mae400-s43` |
| raie | `meanpool-mae450-s43` | 450 | `meanpool-mae450-s43` |
| requin | `meanpool-mae500-s43` | 500 | `meanpool-mae500-s43` |

### Set 2 — Perceiver Q16 ablation (seed 43)

| Host | Run | SSL ep | W&B name |
|------|-----|--------|----------|
| rouget | `perceiverQ16-mae100-s43` | 100 | `perceiverQ16-mae100-s43` |
| roussette | `perceiverQ16-mae200-s43` | 200 | `perceiverQ16-mae200-s43` |
| sole | `perceiverQ16-mae300-s43` | 300 | `perceiverQ16-mae300-s43` |
| thon | `perceiverQ16-mae400-s43` | 400 | `perceiverQ16-mae400-s43` |
| truite | `perceiverQ16-mae500-s43` | 500 | `perceiverQ16-mae500-s43` |

## Provisioning

1. SSL encoders live on gymnote under `checkpoints/track_a/ssl/pretrain/`; copy **one** `ep*.pt` per host via `launch_ft_mae_scaling_s43_fleet.sh copy`.
2. `prep` — `git pull`, `uv sync`, rsync `src/` from coordinator for W&B group/config support.
3. **Canary** — `dataset.max_samples=256`, `training.epochs=1`, log `logs/track_a/canary-<run>_YYYYMMDD.log`; verify `init_from`, W&B URL, step lines.
4. **Overnight** — `launch` (full 50 epochs, no `max_samples` cap).

## Canary acceptance

- `[init_from] loaded` with `encoder-missing=0`
- `[wandb] run started:` with project/group/name
- `[HH:MM:SS] step N/M` lines, ASCII-only, one step per line
- No CUDA OOM / Traceback in first ~2 min

## Related docs

- [`new_ideas_tracka.md`](new_ideas_tracka.md) — official FT recipe (RQ4)
- [`videomaev2_t4_native_300e_pretrain.md`](videomaev2_t4_native_300e_pretrain.md) — SSL encoder provenance
- [`diverse_classifier_heads_post_mae.md`](diverse_classifier_heads_post_mae.md) — Perceiver head (Arch 2)
