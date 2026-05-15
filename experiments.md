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
