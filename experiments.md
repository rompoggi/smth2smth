# Track A — ViT Experiment Plan

## Classifier variants

| ID | Classifier | Status |
|----|-----------|--------|
| **MP** | Mean pool all tokens → Linear (VideoMAE paper) | Implemented |
| **AP** | Attentive Probe (1 learnable query cross-attends all tokens → Linear) | Implemented |

---

## Experiment matrix (MP only)

| # | Name | Who | Description | Config |
|---|------|-----|-------------|--------|
| 1a | **truite** | Romain | No-SSL ViT-B, no class tricks | `track_a_vit_truite` |
| 6a | **rouget** | Romain | No-SSL ViT-B + Class Boosting | `track_a_vit_rouget` |
| 7a | **roussette** | Romain | No-SSL ViT-B + Class Balancing | `track_a_vit_roussette` |
| 8a | **raie** | Romain | No-SSL ViT-B + Class Boosting + Class Balancing | `track_a_vit_raie` |
| 2a | — | Thomas | VideoMAE SSL pre-train → ViT-B fine-tune (75% tube mask) | `track_a_videomae_finetune` |

**Romain** runs truite/rouget/roussette/raie in parallel on 4 machines — all from scratch, no SSL.  
**Thomas** runs the VideoMAE pre-train then 2a fine-tune (sequential).

---

## Run commands

### Thomas — pre-train first (long job, start immediately)
```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.pretrain_videomae \
    experiment=track_a_videomae_pretrain
```

### Romain — run all 4 in parallel on separate machines

**truite**
```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.train \
    experiment=track_a_vit_truite
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
