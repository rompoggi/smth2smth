# Track A — ViT Experiment Plan

## Classifier variants

| ID | Classifier | Status |
|----|-----------|--------|
| **MP** | Mean pool all tokens → Linear (VideoMAE paper) | Implemented |
| **AP** | Attentive Probe (1 learnable query cross-attends all tokens → Linear) | Implemented |

---

## Experiment matrix (MP only)

| # | Name | Who | Description | SSL pre-train | Class Boost | Class Balance | Config |
|---|------|-----|-------------|--------------|-------------|---------------|--------|
| 1a | **truite** | Romain | No-SSL ViT-B baseline | — | — | — | `track_a_videomae_1a` |
| 2a | — | Thomas | VideoMAE → fine-tune | VideoMAE (75% tube mask) | — | — | `track_a_videomae_2a` |
| 6a | **rouget** | Romain | ViT + Class Boosting | — | ✓ | — | `track_a_videomae_6a` |
| 7a | **roussette** | Romain | ViT + Class Balancing | — | — | ✓ | `track_a_videomae_7a` |
| 8a | **raie** | Romain | ViT + Boosting + Balancing | — | ✓ | ✓ | `track_a_videomae_8a` |

**Romain** runs 1a/6a/7a/8a in parallel (no SSL, independent). **Thomas** runs the VideoMAE pre-train then 2a.

---

## Run commands

### Thomas — pre-train first (long job, start immediately)
```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.pretrain_videomae \
    experiment=track_a_videomae_pretrain
```

### Romain — run all 4 in parallel on separate machines

**truite / 1a**
```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.train \
    experiment=track_a_videomae_1a
```

**rouget / 6a**
```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.train \
    experiment=track_a_videomae_6a
```

**roussette / 7a**
```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.train \
    experiment=track_a_videomae_7a
```

**raie / 8a**
```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.train \
    experiment=track_a_videomae_8a
```

### Thomas — pre-train then fine-tune

**Pre-train (start immediately, long job)**
```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.pretrain_videomae \
    experiment=track_a_videomae_pretrain
```

**2a (after pre-training finishes)**
```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.train \
    experiment=track_a_videomae_2a \
    model.init_from=<path/to/videomae_encoder.pt>
```

---

## Fine-tuning hyper-parameters

AdamW · lr 5e-4 (2e-4 for 2a with SSL warm-start) · wd 0.05 · cosine decay to 1e-6  
40 epochs (30 for 2a) · 5 warmup · `drop_path=0.1` · label smoothing 0.1  
RandAug-T(n=4, m=7) · CutMix α=1.0 · **no horizontal flip** · EMA 0.9999

> Layer-wise LR decay (0.75 in the VideoMAE paper) is not yet implemented — compensated by the slightly lower LR in 2a.
