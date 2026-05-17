# HC / mHC ablation on VideoMAE ViT-B

## 1. Purpose

Test whether **Hyper-Connections (HC)** and **Manifold-Constrained Hyper-Connections (mHC)** improve training of a VideoMAE ViT-B fine-tuned on the SSv2 subset. Primary signal is **training stability** (gradient-norm oscillation, loss-curve smoothness); accuracy is secondary.

The experiment is run on **Piranha**. The shared encoder comes from **Sole** (VideoMAE v2, dual-encoder, ViT-B, 200 epochs SSL on `train + test` frames). All three arms fork from the same encoder so that any difference is attributable to the residual structure alone.

## 2. Background — Pre-Norm baseline

In a Pre-Norm Transformer block with sublayer `F ∈ {Attn, MLP}` (each with its own LayerNorm `Norm`):

```python
# Pre-Norm residual (baseline)
def block(x):
    x = x + Attn(Norm_attn(x))
    x = x + MLP(Norm_mlp(x))
    return x
```

Pre-Norm is the standard for ViTs. It is stable at moderate depth but is known to suffer from a **gradient-vanishing ↔ representation-collapse seesaw** (HC paper, §2): scale up depth and gradients vanish; reduce normalization and representations collapse.

## 3. Method — Hyper-Connections (HC)

HC widens the residual stream into `n` parallel "hyper-hidden" vectors mixed by a learnable matrix at each sublayer. With expansion rate `n` and hidden dim `d`, each block maintains a matrix `H ∈ R^{n×d}`.

For each sublayer F, HC defines four learnable objects:
- `α_pre ∈ R^n` — depth-mix weights that read a single vector from `H` to feed F.
- `M ∈ R^{n×n}` — static mixing matrix that propagates the n streams forward.
- `β ∈ R^n` — write weights for the sublayer output back into `H`.
- `α_out ∈ R^n` — read weights at the end of the stack to collapse `H` back to one vector.

**Identity-equivalent initialization** (paper §3.4): `M = I_n`, `α_pre = β = α_out = e_0` (one-hot at index 0). At step 0 this is exactly Pre-Norm, so HC is a **zero-risk capacity injection** — at worst it stays identity, at best it learns useful cross-stream routing.

### 3.1 Pseudocode (Static HC, the variant we test)

```python
# Static HC, expansion rate n, per sublayer F
class HCBlock(nn.Module):
    def __init__(self, d, n, sublayer):
        self.n, self.F = n, sublayer
        # identity init: M = I, alpha_pre = beta = e_0
        self.M        = nn.Parameter(torch.eye(n))
        self.alpha_pre = nn.Parameter(F.one_hot(torch.tensor(0), n).float())
        self.beta      = nn.Parameter(F.one_hot(torch.tensor(0), n).float())

    def forward(self, H):                     # H: (B, n, T, d)
        h_pre = einsum("n,bntd->btd", self.alpha_pre, H)
        y     = self.F(layer_norm(h_pre))     # (B, T, d)
        H     = einsum("ij,bjtd->bitd", self.M, H) \
              + einsum("n,btd->bntd", self.beta, y)
        return H

# wrapper around the whole ViT
def forward_vit(x):
    H = x.unsqueeze(1).expand(-1, n, -1, -1)  # H[i] = x for all i
    for block in blocks:                       # each block: attn-HC then mlp-HC
        H = block.hc_attn(H)
        H = block.hc_mlp(H)
    return einsum("n,bntd->btd", alpha_out, H) # collapse back to one stream
```

**Extra params at n=4 for ViT-B (24 sublayers, d=768):**
- `M`: n² = 16 per sublayer → 384 total
- `α_pre, β`: 2n = 8 per sublayer → 192 total
- `α_out`: n = 4
- **Total ≈ 600 scalars on top of 87M — negligible.**

We pick **Static HC (SHC)** over Dynamic HC (DHC). DHC makes `α_pre, β` input-dependent (Linear(d→n)) and is the variant favored at LLM scale; at our scale its 150k extra parameters add variance without clear upside (HC paper Table 2 / Appendix F).

## 4. Method — Manifold-Constrained HC (mHC)

mHC replaces HC's unconstrained `M` with the projection of a raw learnable matrix `M̃` onto the **Birkhoff polytope** (doubly stochastic matrices) via **Sinkhorn-Knopp**. This restores an identity-mapping-like property: doubly stochastic matrices have spectral radius ≤ 1, preventing forward-signal blowup with depth.

```python
def sinkhorn_knopp(M_raw, K=3, tau=1.0):
    # cast to fp32 for numerical stability under AMP bf16
    M = (M_raw / tau).float().exp()
    for _ in range(K):
        M = M / M.sum(dim=1, keepdim=True)   # row-normalize
        M = M / M.sum(dim=0, keepdim=True)   # column-normalize
    return M

class mHCBlock(HCBlock):
    def __init__(self, d, n, sublayer, K=3, tau=1.0):
        super().__init__(d, n, sublayer)
        self.K, self.tau = K, tau
        # raw learnable matrix, init so SK(M_raw) ≈ I
        self.M_raw = nn.Parameter(large_diagonal_init(n))
        del self.M  # M is now computed on the fly

    def forward(self, H):
        M = sinkhorn_knopp(self.M_raw, K=self.K, tau=self.tau).to(H.dtype)
        h_pre = einsum("n,bntd->btd", self.alpha_pre, H)
        y     = self.F(layer_norm(h_pre))
        H     = einsum("ij,bjtd->bitd", M, H) \
              + einsum("n,btd->bntd", self.beta, y)
        return H
```

**SK iterations:** paper uses K=20 at LLM scale. We use **K=3** at n=4 — the projection converges to <1e-3 deviation from doubly-stochastic at this size and the cost (2n² ops per iteration → 96 ops per block) is essentially free.

## 5. Why test at our scale — honest framing

The HC and mHC stability claims are anchored at **LLM scale** (3B–27B params, 40+ blocks). At ViT-B (12 blocks × n=4) the forward-signal blowup that mHC fixes is unlikely to manifest. Expected accuracy delta vs Pre-Norm is **−0.5 to +1.0 pt** with non-trivial probability of zero effect.

We run the ablation anyway because:
1. **<1% parameter and compute overhead** — the cost is small.
2. **Identity-equivalent init** — HC cannot regress below Pre-Norm at step 0, so the baseline arm is protected.
3. **HC paper Appendix F** shows small-but-positive gains on ImageNet (ViT-B/L) and DiT generation — there is some vision precedent.
4. **mHC's Sinkhorn projection acts as a structural regularizer** on the residual mixing matrix; in a low-data regime (~45k clips) this may help even if the stability fix itself is unnecessary.

## 6. Pretrained base

| Item                | Value                                                          |
|---------------------|----------------------------------------------------------------|
| Run host            | Sole                                                           |
| Model               | VideoMAE v2 (dual encoder), ViT-B                              |
| SSL data            | `train + test` frames (closed-world, val excluded)             |
| SSL epochs          | 200                                                            |
| Per-epoch wall      | ≈2.7 min observed (15 ep in 40 min)                            |
| ETA for full SSL    | ≈9 h                                                           |
| Checkpoint location | `checkpoints/track_a/ssl/sole_encoder.pt` (encoder)            |
| Decoder saved       | yes, every epoch — crash recovery does not reset MAE objective |

**Why this base, not E5 or a from-scratch ViT-S:**
- VideoMAE v2 dual-encoder + 200 ep > vanilla MAE 100 ep — a credible baseline.
- ViT-B's wider residual stream (d=768) gives HC's cross-stream mixing more capacity than ViT-S (d=384). Depth is unchanged (both 12 blocks), so this does not enlarge the stability signal — but it strengthens the absolute baseline and makes the report more publishable.
- Decoder is checkpointed per epoch, so a Sole restart does not reset MAE pretraining.

## 7. Experiment design

| Arm | Name              | Residual structure                       | Encoder init           |
|-----|-------------------|------------------------------------------|------------------------|
| (a) | `baseline_prenorm`| Pre-Norm                                 | `sole_encoder.pt`      |
| (b) | `shc_n4`          | Static HC, n=4, identity init            | `sole_encoder.pt`      |
| (c) | `mhc_n4_sk3`      | mHC static, n=4, K=3 SK iters, τ=1.0     | `sole_encoder.pt`      |

**3 seeds per arm**, seed-paired across arms, for **9 FT runs total**.

**Run order:** (a) → (b). Launch (c) only if (b)'s mean ≥ (a)'s mean − 0.3 pt. If HC clearly hurts, mHC is even less likely to help — abort to save wall time.

### 7.1 FT recipe (identical for all three arms)

Inherit `track_a_ssl_finetune_*.yaml` (champion FT recipe). Key knobs:

```yaml
ft:
  epochs: 60
  opt: adamw
  blr: 5e-4
  opt_betas: [0.9, 0.999]
  weight_decay: 0.05
  layer_decay: 0.75
  drop_path: 0.1
  dropout: 0.5
  attentive_head:
    heads: 4
  frame_mixup_alpha: 5
  randaug: temporal_plus_n2_m9
  ema_decay: 0.999
  eff_batch_size: 64        # may need grad-accum for ViT-B on a 3090
  precision: bf16

# HC-only additions
hc:
  variant: static            # SHC, not DHC
  n: 4
  init: identity             # M=I, alpha_pre=beta=e_0
  scalar_wd: 0.0             # treat HC scalars like LayerNorm gains

# mHC-only additions
mhc:
  sk_iters: 3
  tau: 1.0
  sk_dtype: fp32             # SK exp/normalize in fp32 even under AMP bf16
```

**Hard rules:**
- HC scalars (`M`, `α_pre`, `β`, `α_out`) get `weight_decay=0` and are excluded from `layer_decay` (treat like LN gains, per BEiT/MAE convention).
- HC scalars are included in the EMA buffer.
- All three arms use identical seeds and identical data ordering. Set `dataset.use_official_val=true`, `dataset.include_val_in_train=false`.
- TTA at submit time: ViT scales `[0.857, 1.0, 1.143]` + flip (already in finetune YAML).

### 7.2 Configs to add

Three new files under `configs/experiment/`:

- `track_a_hc_ablation_baseline.yaml` — Pre-Norm FT from `sole_encoder.pt`.
- `track_a_hc_ablation_shc.yaml` — SHC n=4 FT from `sole_encoder.pt`.
- `track_a_hc_ablation_mhc.yaml` — mHC n=4 SK=3 FT from `sole_encoder.pt`.

Each takes a `seed` override (`seed=42`, `seed=43`, `seed=44`).

## 8. Logging plan

The stability claim is a **variance claim** and requires high-frequency logging.

### Per-step (every 100 steps minimum, ideally every step):
- `train/loss` — instantaneous, not running average
- `train/grad_norm_global` — pre-clip global L2 norm
- `train/grad_norm_per_block` — L2 norm per transformer block (for depth analysis)
- `train/lr` — for sanity
- `train/hc/M_max_abs`, `train/hc/M_off_diag_mass` — only for arms (b)/(c); track how far M drifts from identity
- `train/mhc/sk_doubly_stochastic_dev` — only arm (c); max deviation of SK(M_raw) from doubly-stochastic, sanity check

### Per-epoch:
- `train/avg_loss`, `train/top1`
- `val/loss`, `val/top1`, `val/top5`
- `val_ema/loss`, `val_ema/top1`, `val_ema/top5`
- `wall/epoch_seconds`

### Run metadata:
- Seed, git SHA, encoder hash, arm name, full config.

Use TensorBoard or W&B; CSV-dump per-step grad norms so plotting code does not depend on the dashboard.

## 9. Plots (one per metric, averaged across 3 seeds)

For each metric, **one figure** with three curves (Pre-Norm, HC, mHC). Plot **mean ± std** across seeds as a shaded band. X-axis is **epoch** (or step for grad-norm plots).

Mandatory plots:

1. **Train loss vs epoch** — mean ± std band.
2. **Val Top-1 vs epoch** (raw and EMA, two subplots).
3. **Val loss vs epoch.**
4. **Global grad norm vs step** — the headline stability plot. Smoothing window ~50 steps to see oscillation pattern.
5. **Per-block grad norm vs step** — heatmap or facet plot, one row per block. Helps see whether HC concentrates gradient flow at specific depths.
6. **HC mixing matrix drift** — `||M − I||_F` vs epoch for arms (b) and (c). Shows whether and how fast each method learns away from identity.

Optional (if time permits):
- **Loss oscillation metric** — rolling std of per-step train loss over a 500-step window. Quantifies the smoothness claim numerically.
- **Per-seed scatter** of final EMA val Top-1 (3 seeds × 3 arms = 9 points) with means marked.

## 10. Success / closed criteria

Defined upfront to avoid post-hoc rationalization.

- **HC success:** arm (b) honest EMA val Top-1 ≥ arm (a) + 0.5 pt averaged over 3 seeds.
- **mHC success:** arm (c) > arm (b) by 0.3 pt **AND** arm (c) > arm (a) by 0.5 pt.
- **Stability success (secondary):** arm (b) and/or (c) show ≥20% reduction in rolling std of per-step grad norm vs arm (a), measured over the second half of training.
- **Closed criterion:** if neither (b) nor (c) beats (a) by ≥0.3 pt on accuracy AND neither shows the 20% grad-norm-std reduction, HC/mHC are dead at this scale. **Do not run DHC or n=8 follow-ups.**

## 11. Wall-clock estimate (Piranha, 1× RTX 3090)

| Phase                              | Wall  |
|------------------------------------|-------|
| Wait for Sole's ViT-B SSL to finish| ~9 h  |
| FT arm × 1 seed (ViT-B, 60 ep)     | ~3–4 h |
| 3 arms × 3 seeds = 9 FT runs       | ~30 h |
| Plot generation + report writeup   | ~1 h  |
| **Total after SSL finishes**       | ~30 h |

Fits roughly two overnight windows. If VRAM forces grad-accum, expect a 10–20% wall increase.

## 12. Do not

- Run on a Sole encoder snapshot that has not seen the full 200 epochs (unless arm (a) sanity check at epoch ~100 says it is already strong enough).
- Use DHC, or n>4 — out of scope; only revisit if (b)/(c) win at n=4.
- Mix seeds across arms — seed-paired comparison is essential for the variance story.
- Apply weight decay to HC scalars.
- Forget to cast the Sinkhorn `exp`/normalize block to fp32 under AMP bf16 (NaN risk).

## References

- Zhu et al. *Hyper-Connections.* ICLR 2025. arXiv:2409.19606
- Xie et al. *Manifold-Constrained Hyper-Connections.* 2025. arXiv:2512.24880
- HC reference implementation pattern: `tokenbender/mHC-manifold-constrained-hyper-connections`
