# Track A — ViT Experiment Plan

## Classifier variants

| ID | Classifier | Status |
|----|-----------|--------|
| **MP** | Mean pool all tokens → Linear (VideoMAE paper) | Implemented |
| **AP** | Attentive Probe (1 learnable query cross-attends all tokens → Linear) | Implemented |

---

## Experiment matrix (MP only)

| # | Who | Description | SSL pre-train | Class Boost | Class Balance | Config |
|---|-----|-------------|--------------|-------------|---------------|--------|
| 1a | Romain | No-SSL ViT-B baseline | — | — | — | `track_a_videomae_1a` |
| 2a | Thomas | VideoMAE → fine-tune | VideoMAE (75% tube mask) | — | — | `track_a_videomae_2a` |
| 6a | Romain | ViT + Class Boosting | best of 1a/2a | ✓ | — | `track_a_videomae_6a` |
| 7a | Thomas | ViT + Class Balancing | best of 1a/2a | — | ✓ | `track_a_videomae_7a` |
| 8a | first free | ViT + Boosting + Balancing | best of 1a/2a | ✓ | ✓ | `track_a_videomae_8a` |

**Dependency**: 6a/7a/8a wait for 1a vs 2a comparison. Add `model.init_from=<ckpt>` if 2a wins.

---

## Run commands

### Thomas — pre-train first (long job, start immediately)
```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.pretrain_videomae \
    experiment=track_a_videomae_pretrain
```

### Romain — 1a (no dependency)
```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.train \
    experiment=track_a_videomae_1a
```

### Thomas — 2a (after pre-training finishes)
```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.train \
    experiment=track_a_videomae_2a \
    model.init_from=<path/to/videomae_encoder.pt>
```

### Romain — 6a (after 1a/2a comparison)
```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.train \
    experiment=track_a_videomae_6a
# If 2a > 1a, add: model.init_from=<path/to/videomae_encoder.pt>
```

### Thomas — 7a (after 1a/2a comparison)
```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.train \
    experiment=track_a_videomae_7a
# If 2a > 1a, add: model.init_from=<path/to/videomae_encoder.pt>
```

### 8a (whoever is free)
```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.train \
    experiment=track_a_videomae_8a
# If 2a > 1a, add: model.init_from=<path/to/videomae_encoder.pt>
```

---

## Fine-tuning hyper-parameters

AdamW · lr 5e-4 (2e-4 for 2a with SSL warm-start) · wd 0.05 · cosine decay to 1e-6  
40 epochs (30 for 2a) · 5 warmup · `drop_path=0.1` · label smoothing 0.1  
RandAug-T(n=4, m=7) · CutMix α=1.0 · **no horizontal flip** · EMA 0.9999

> Layer-wise LR decay (0.75 in the VideoMAE paper) is not yet implemented — compensated by the slightly lower LR in 2a.
