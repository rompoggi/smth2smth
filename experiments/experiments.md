# Track A — ViT Experiment Plan

## Classifier variants

| ID | Classifier | Status |
|----|-----------|--------|
| **MP** | Mean pool all tokens → Linear (VideoMAE paper) | Implemented |
| **AP** | Attentive Probe (1 learnable query cross-attends all tokens → Linear) | Implemented |

---

## Experiment matrix (supervised ViT from scratch — **all failed**; use SSL)

| # | Name | Who | Description | Config |
|---|------|-----|-------------|--------|
| 1a | **truite** | Romain | No-SSL ViT-B, mean pool, no class tricks | `track_a_vit_truite` |
| 1b | **truite AP** | Romain | No-SSL ViT-B, attentive probe (same recipe as 1a); **stopped**, no gain vs MP | `track_a_vit_truite_ap` |
| — | **sardine** | Romain | No-SSL ViT-S, mean pool, large batch + heavy RandAug | `track_a_vit_sardine` |
| 6a | **rouget** | Romain | No-SSL ViT-B + Class Boosting | `track_a_vit_rouget` |
| 7a | **roussette** | Romain | No-SSL ViT-B + Class Balancing | `track_a_vit_roussette` |
| 8a | **raie** | Romain | No-SSL ViT-B + Class Boosting + Class Balancing | `track_a_vit_raie` |
| 2a | — | Thomas | VideoMAE SSL pre-train → ViT-B fine-tune (75% tube mask) | `track_a_videomae_finetune` |

**Conclusion:** supervised VideoMAE-style ViT from random init does not learn on this dataset at Track-A scale (\(\sim 45\)k clips). **Next step:** `track_a_videomae_pretrain` → `track_a_videomae_finetune` with `model.init_from`.

---

## Run commands

### Thomas — pre-train first (long job, start immediately)
```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.pretrain_videomae \
    experiment=track_a_videomae_pretrain
```

### Romain — run all 4 in parallel on separate machines

**truite** (mean pool)
```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.train \
    experiment=track_a_vit_truite
```

**truite AP** (attentive probe, same hyperparams as truite)
```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.train \
    experiment=track_a_vit_truite_ap
```

**rouget**
```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.train \
    experiment=track_a_vit_rouget
```

**roussette**
```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.train \
    experiment=track_a_vit_roussette
```

**raie**
```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.train \
    experiment=track_a_vit_raie
```

### Thomas — pre-train then fine-tune

**Pre-train (start immediately, long job)**
```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.pretrain_videomae \
    experiment=track_a_videomae_pretrain
```

**Fine-tune (after pre-training finishes)**
```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.train \
    experiment=track_a_videomae_finetune \
    model.init_from=<path/to/videomae_encoder.pt>
```

---

## Fine-tuning hyper-parameters

AdamW · lr 5e-4 (2e-4 for 2a with SSL warm-start) · wd 0.05 · cosine decay to 1e-6  
40 epochs (30 for 2a) · 5 warmup · `drop_path=0.1` · label smoothing 0.1  
RandAug-T(n=4, m=7) · CutMix α=1.0 · **no horizontal flip** · EMA 0.9999

> Layer-wise LR decay (0.75 in the VideoMAE paper) is not yet implemented — compensated by the slightly lower LR in 2a.

---

## SSL night sweep (slots 8–16, `SSL.md`)

VideoMAE pretrain on **train + val + test** (unlabeled) → supervised fine-tune with **attentive probe** (`track_a_ssl_pretrain_<fish>` / `track_a_ssl_finetune_<fish>`). One codename per VM; configs under `configs/experiment/`.

| Slot | Codename | Who | Status | Notes |
|------|----------|-----|--------|-------|
| 8 | **barbeau** | Thomas | **Not run** | — |
| 9 | **truite** | Romain | **Pretrain only** | Phase 1 done; Phase 2 not started (see below) |
| 10 | **roussette** | Romain | **Done** | `seed` drift 123→42; pretrain `bs=8` (see below) |
| 11 | **rouget** | Romain | **Done** (see below) | Long MAE (`pretrain.epochs=200`) |
| 12 | **raie** | Romain | **Done** | 6-head probe; best EMA val **34.80%** |
| 13 | **sole** | Romain | **Not run** (this host) | 8-head probe; GPU deferred (see below) |
| 14 | **thon** | Romain | **Not run** (this host) | MAE `mask_ratio=0.85`; never launched |
| 15 | **piranha** | Romain | **Done** | FT **40** ep (vs 30 default); best EMA val **35.60%** |
| 16 | **murene** | Romain | **Not run** | Dataset download blocked (24h user limit exceeded) |

### Slot 9 — **truite** (documented)

**Configs:** `track_a_ssl_pretrain_truite` (ViT-S baseline, `seed=42`, MAE 100 epochs) → `track_a_ssl_finetune_truite` (attentive probe, 4 heads, 30 FT epochs) — **finetune not launched**.

**Phase 1 — pretrain:** **Done** 100/100. Log: `logs/ssl_truite_pretrain.log`. Encoder: `checkpoints/track_a/ssl/truite_encoder.pt` (final epoch weights; checkpoint rewritten each epoch). MAE avg loss: 0.851 (ep 1) → 0.279 (ep 50) → **0.256** (ep 100). ~58 651 unlabeled clips. Hydra config matches YAML; no `include_val_in_train` or resume overrides in the successful run.

**Phase 2 — fine-tune:** **Not run.** No `logs/ssl_truite_finetune.log`, no `truite_ft.pt`. No train/val top-1 or submission.

**Config drift:** None on the successful pretrain (launcher may have used `.venv` Python rather than `uv` in `PATH` — not recorded in log). Phase 2 pending per `SSL.md` §4.

**Next step:** `experiment=track_a_ssl_finetune_truite` with `model.init_from` → `truite_encoder.pt`.

### Slot 10 — **roussette** (documented)

**Configs:** `track_a_ssl_pretrain_roussette` (ViT-S, YAML `seed=123`, MAE 100 ep) → `track_a_ssl_finetune_roussette` (attentive probe, 4 heads, 30 FT ep).

**Phase 1 — pretrain:** **Done** 100/100 (~7 h). Log: `logs/ssl_roussette_pretrain.log`. Encoder: `checkpoints/track_a/ssl/roussette_encoder.pt`. MAE avg loss: 0.8429 (ep 1) → **0.2510** (ep 100). Encoder overwritten each epoch; on-disk file = epoch 100.

**Phase 2 — fine-tune:** **Done** 30/30 (~2.5 h). Log: `logs/ssl_roussette_finetune.log`. Official val only. Final train top-1 **41.44%**; final val **33.62%** live / **33.31%** EMA. **Best saved: 33.61% EMA (epoch 27)** → `roussette_ft.pt`. `init_from` loaded 149 encoder tensors, 0 missing.

**Config drift:** `pretrain.batch_size` **16 → 8** (CLI). Resolved **`seed=42`** in log vs **123** in YAML (likely root `config.yaml` overriding experiment). No `include_val_in_train`, no resume. bf16 `scatter_` fix required for pretrain to run (no error in final log).

**Submission:** None yet (`roussette_ft.pt` not submitted).

### Slot 13 — **sole** (documented)

**Configs:** `track_a_ssl_pretrain_sole` (ViT-S, MAE 100 epochs) → `track_a_ssl_finetune_sole` (`head_num_heads=8` vs truite/rouget=4, raie=6).

**Phase 1 — pretrain:** **Not run** on `/Data/romain.poggi/smth2smth`. No `logs/ssl_sole_pretrain.log`, no `sole_encoder.pt`. Planned YAML exists; no Hydra log to compare (config drift **N/A**).

**Phase 2 — fine-tune:** **Not run.** No finetune log, no `sole_ft.pt`, no train/val top-1 or submission.

**Note:** Launch was deferred when GPU 0 was fully utilized by another job (~18.6/24.6 GiB). Do not confuse with supervised `track_a_vit_*` logs (`logs/raie.log`, etc.) — those are mean-pool ViT runs, not the SSL sweep.

**Next step:** When GPU is free, run `SSL.md` §4 with `CODENAME=sole`; if artifacts exist on another VM, copy `logs/ssl_sole_*.log` here for metrics.

### Slot 14 — **thon** (documented)

**Configs:** `track_a_ssl_pretrain_thon` (ViT-S, `mask_ratio=0.85` vs default 0.75) → `track_a_ssl_finetune_thon` (attentive probe, 4 heads, 30 FT ep).

**Phase 1 — pretrain:** **Not run** on this VM. No `logs/ssl_thon_pretrain.log`, no `thon_encoder.pt`; `checkpoints/track_a/ssl/` may be absent until first sweep job creates it.

**Phase 2 — finetune:** **Not run.** No finetune log, no `thon_ft.pt`, no train/val top-1 or submission.

**Config drift:** **N/A** (no launch; YAML on disk matches `SSL.md` §7). Repo sync ~2026-05-16 01:12; sweep command from §4 apparently not executed.

**Next step:** `mkdir -p logs checkpoints/track_a/ssl`, then Phase 1/2 per `SSL.md` with `CODENAME=thon`.

### Slot 12 — **raie** (documented)

**Configs:** `track_a_ssl_pretrain_raie` (ViT-S, MAE 100 ep) → `track_a_ssl_finetune_raie` (`head_num_heads=6`).

**Phase 1 — pretrain:** **Done** 100/100 after one failed attempt (`scatter_` dtype under AMP; fixed in `video_mae.py`). Logs: `ssl_raie_pretrain.crash.log`, then `ssl_raie_pretrain.log`. Encoder: `raie_encoder.pt`. MAE loss: 0.851 → **0.254** (ep 100). ~3h40 wall.

**Phase 2 — finetune:** **Done** 30/30 via `chain_raie.sh` (~1h27). Log: `ssl_raie_finetune.log`. Official val only (44 993 train / 6 745 val). Train top-1 **44.15%** (ep 30); best **EMA val 34.80%** (ep 29) → `raie_ft.pt`. `init_from`: 149 tensors OK.

**Config drift:** Chained launcher vs manual §4; pretrain retry after code fix; `uv` path under `nohup`. Hydra hyperparams match YAML. **Not part of sweep:** `ssl_raie_finetune_resume_official_val.log` — OOM at startup, 0 epochs.

**Submission:** None.

### Slot 15 — **piranha** (documented)

**Configs:** `track_a_ssl_pretrain_piranha` (ViT-S, MAE 100 ep) → `track_a_ssl_finetune_piranha` (**40** FT epochs, attentive probe 4 heads).

**Phase 1 — pretrain:** **Done** 100/100 (~7 h). Log: `logs/ssl_piranha_pretrain.log`. Encoder: `piranha_encoder.pt`. Final MAE avg loss **0.2528**. Hydra matches YAML (`bs=16`, mask 0.75).

**Phase 2 — fine-tune:** **Done** 40/40 (no early stop). Log: `logs/ssl_piranha_finetune.log`. Train top-1 **49.54%** (ep 40); best **EMA val 35.60%** (ep 32) → `piranha_ft.pt`. Final ep 40 val: live **34.65%**, EMA **34.94%**. `piranha_ft.last.pt` = epoch 40. `init_from`: 149 tensors, 0 missing.

**Config drift:** Chained `nohup` pipeline vs two separate §4 jobs (operational only). Pretrain/finetune Hydra bodies match YAML. No `include_val_in_train`, no resume.

**Submission:** None.

### Slot 16 — **murene** (documented)

**Configs:** `track_a_ssl_pretrain_murene` → `track_a_ssl_finetune_murene` (FT `lr=3e-4` per `SSL.md` §7).

**Status:** **Not run.** The dataset could not be downloaded on the assigned VM — **storage/user quota limit exceeded within 24h** — so neither Phase 1 nor Phase 2 was launched. No `ssl_murene_*` logs, encoders, or metrics.

**Config drift:** N/A.

**Next step:** Retry dataset fetch after quota reset, then run `SSL.md` §4 with `CODENAME=murene`.

### Slot 11 — **rouget** (documented)

**Configs:** `track_a_ssl_pretrain_rouget` (ViT-S, MAE 200 epochs, `mask_ratio=0.75`) → `track_a_ssl_finetune_rouget` (attentive probe, 4 heads, 30 FT epochs, `lr=2e-4`).

**Phase 1 — pretrain:** Finished. Encoder: `checkpoints/track_a/ssl/rouget_encoder.pt`. Log: `logs/ssl_rouget_pretrain.log`.

**Phase 2a — fine-tune (honest split):** Official val only (`use_official_val=true`, train = `data/train` only). Finished 30/30 epochs. Best **EMA val top-1: 34.96%** (epoch 30). Log: `logs/ssl_rouget_finetune.log`. Checkpoint: `checkpoints/track_a/ssl/rouget_ft.pt`.

**Phase 2b — continued fine-tune (mistake):** Resumed from `rouget_ft.last.pt` with `dataset.include_val_in_train=true` (train + val clips, still evaluating on the same val set). Stopped manually at epoch 44/60. Best val top-1 on log reached **~60%** — **not a meaningful metric** (labels from val were in the training set; no held-out validation). Log: `logs/ssl_rouget_finetune_trainval.log`.

**Submission:** `submissions/track_a_rouget_ssl.csv` from `rouget_ft.pt` after Phase 2b (flip-only TTA, `tta_scales=[1.0]`; VideoMAE incompatible with multi-scale TTA). **Public Kaggle top-1: 36%.**

**Takeaways:**

1. **Honest val (~35%) matches the leaderboard (~36%)** — SSL ViT-S + attentive probe is in the same ballpark as our previous Track-A CNN best (~37.7% public), not a breakthrough.
2. **Training on validation data without keeping val held out** inflated internal val to ~60% but **did not improve Kaggle**; we likely **overfit** the official val distribution. A cleaner rerun should keep `include_val_in_train=false`, pick the Phase 2a checkpoint, and only add val to training for a **final** retrain if we accept no unbiased val metric for model selection.
