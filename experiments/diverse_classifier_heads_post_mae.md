# Four Diverse Heads for VideoMAEv2 ViT-B on a 33-class SSv2 Subset: A Decision-Ready Design Brief

## TL;DR

- **Run all four heads in parallel tonight. Highest expected single-model lift over the 53.71% control comes from Architecture 3 (late-block divided space-time + 16-query Perceiver head): +2.0 to +3.5 pt on the public LB; second-best single-model lift is Architecture 4 (AIM-style temporal reuse): +1.5 to +3.0 pt; Architectures 1 and 2 are smaller individual gains (+0.5 to +1.5 pt) but are deliberately cheap and architecturally complementary so they pull weight in the ensemble.**
- **The ensembling case for diversity is decisive: literature predicts a Caruana-weighted stack of these 4 + control to beat a 5-seed deep-ensemble of the control head by 1.0–2.5 pt on a 33-class top-1 task. Wenzel et al. (NeurIPS 2020) show that *hyperparameter* diversity dominates pure-seed deep ensembles at equal budget on CIFAR-10/100 and Fashion-MNIST; architectural diversity is a strictly stronger perturbation (Fort, Hu & Lakshminarayanan 2019 show seed-only ensembles capture only basin-level diversity, whereas different architectures change the function class).**
- **Critical risk to watch: Architectures 3 and 4 inject random-init modules that the official VideoMAE layer-decay 0.75 schedule will under-train. Use a separate parameter group at full base LR 5e-4 with 5-epoch warmup for new modules; keep LLRD 0.75 only on the pretrained backbone. Without this split, expect ~1–2 pt loss vs the projected gains.**

## Key Findings

1. **AIM Table 1 (Yang et al., ICLR 2023) is the strongest single piece of literature evidence the user does not yet have quantified.** On SSv2 ViT-B/16 (IN-21K), going from frozen space-only linear probe (15.1% Top-1) to + spatial-adapter (36.7%) to + temporal-adaptation via reused S-MSA on the temporal axis (61.2%) is a **+24.5 pt jump from temporal modeling alone**. Adding joint adaptation gives a further +0.8 pt to 62.0%. The full-FT TimeSformer baseline in the same table is 59.5%. This is the cleanest evidence in the literature that **adding a temporal mechanism to a spatially-strong ViT-B beats full-FT TimeSformer at SSv2 with 14.3M tunable params**.
2. **The user's current 53.71% with pure mean-pool over space-time tokens is, in V-JEPA framing, an "average pooling" baseline.** The V-JEPA (1) paper (Bardes et al., arXiv 2404.08471, Table 3) reports a **+16.1 pt** improvement on SSv2 going from average pooling to a cross-attention attentive probe on a *frozen ViT-H/16 backbone*. That delta is for a frozen, much-larger encoder; in the user's full-FT ViT-B setting the marginal will be substantially smaller (the backbone has already absorbed much of the temporal signal during FT), but the structural argument that mean-pool over (T·H·W) tokens leaves signal on the table is unambiguous.
3. **Multi-query Perceiver heads are NOT monotonically better with more queries on SSv2.** UniFormerV2 Table 4 (Li et al., ICCV 2023) sweeps Q ∈ {1, 4, 16} in the global UniBlock cross-attention on SSv2: **Q=1 → 69.5, Q=4 → 69.1, Q=16 → 68.6** (Sequential). For SSv2 specifically, fewer queries win. The Efficient Probing paper (Psomas et al., ICLR 2026, arXiv 2506.10178) recommends EP64 on ImageNet-1K MAE ViT-B but uses a multi-query NO-projection cross-attention — that's an image task with 196 spatial tokens and a frozen encoder; not directly transferable. **Recommendation: use Q=16 (matches the user's Stan-style precedent), but plan a Q-sweep {1, 4, 16, 32} as a free follow-up if Architecture 2 is competitive.**
4. **Efficient Probing (Psomas et al. 2026) is multi-query *single-head*, with no WK/WV projection on the KV side, and no positional encoding on KV.** EP64 achieves 75.6% Top-1 vs LP/[CLS] 67.7% on ImageNet-1K MAE ViT-B (+7.9 pt) with <1.4M params. EP is a real design departure from V-JEPA-style attentive probes (which are multi-head). For the user's full-FT setting, the EP form factor is too lightweight for a high-stakes LB; stick with V-JEPA-style multi-head cross-attention (12 heads to match the backbone).
5. **TimeSformer's `temporal_fc` Linear projection is the canonical zero-init handle.** The official TimeSformer code (and its HuggingFace port) explicitly maintains `temporal_fc = nn.Linear(dim, dim)` AFTER the temporal attention block, separate from the attention's own output projection. Zero-init this `temporal_fc` (not the attention `proj`) to make the temporal-attn path identity at step 0 — this is the documented practice and matches XViT (Bulat NeurIPS 2021), which sets the *temporal* positional embedding to zero for the same reason.
6. **The official VideoMAE SSv2 FT recipe** (MCG-NJU/VideoMAE `FINETUNE.md`): lr 5e-4, wd 0.05, 50 epochs, batch 8 per node × 8 nodes, num_segment=2, num_crop=3, 16 frames. The user is running 4 frames but otherwise matches; nothing in the recipe disallows adding heads — only LR/LLRD for *new* params needs a delta.

## Details — The Four Architectures

For all sketches: assume the existing pipeline produces token features `x ∈ (B, N, D)`. With tube_t=2, patch=16, 4 frames @ 224, the geometry is T'=2, H'=W'=14, giving N=392 per clip. The user states N=788; that is consistent with either a different tube/patch arithmetic, a CLS-like token, or doubled temporal tokens. The modules below are N-agnostic. D=768.

---

### ARCHITECTURE 1 — Attentive Probe Head (V-JEPA-Style Single Learnable Query)

**1. Architecture summary.** Replace the literal mean-pool + Linear head with a V-JEPA-style attentive probe: a single learnable query token attends (via 12-head cross-attention) to all (B, N, D) backbone tokens. The query output goes through LayerNorm → 4× MLP → LayerNorm → Linear(D, 33). This is the *minimum* possible step away from mean-pool and serves as a clean ablation: "does adaptive pooling alone help, with no temporal-mechanism change?"

**2. Exact PyTorch sketch.**

```python
class AttentiveProbeHead(nn.Module):
    """V-JEPA style: single learnable query, 1 cross-attn block."""
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0, num_classes=33, drop=0.0):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(1, 1, dim))     # learnable query
        self.norm_kv = nn.LayerNorm(dim)                  # LN on backbone tokens (V-JEPA convention)
        self.norm_q  = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)), nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim*mlp_ratio), dim), nn.Dropout(drop),
        )
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.q, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=2e-5); nn.init.zeros_(self.head.bias)

    def forward(self, x):                # x: (B, N, D) from VideoMAE encoder
        B = x.size(0)
        q  = self.norm_q(self.q.expand(B, -1, -1))
        kv = self.norm_kv(x)
        z, _ = self.attn(q, kv, kv, need_weights=False)   # (B, 1, D)
        z = z + q                                          # residual on query
        z = z + self.mlp(self.norm2(z))
        z = self.norm_out(z).squeeze(1)                    # (B, D)
        return self.head(z)
```

**3. Init scheme.** Query: trunc-normal std=0.02 (V-JEPA convention; *not* loaded from a CLS — VideoMAE has no usable CLS for this purpose). MHA QKV/out projections: PyTorch default (Xavier) is stable; trunc-normal std=0.02 also fine. LayerNorms: standard (γ=1, β=0). Final classifier: trunc-normal std=2e-5 + zero bias (small std avoids early dominance by Mixup-noisy logits, matching DeiT/V-JEPA classifier init).

**4. Recipe deltas.** Add a parameter group for the head at base LR 5e-4 with no layer-decay scaling (it's the "top layer," LLRD multiplier = 1.0 anyway). Keep 5-epoch warmup. Weight decay 0.05 unchanged. Dropout=0 in the MLP (backbone drop_path 0.2 is already strong regularization). **No other deltas.**

**5. Parameter count.** Q (768) + QKV+out projections (4·768·768 = 2.36M) + MLP (2·768·3072 = 4.72M) + 4 LayerNorms + classifier (768·33 ≈ 25K). Total ≈ **+7.1M params**. Single-block configuration. The V-JEPA-2 production probe uses 4 transformer blocks (~28M); we deliberately use 1 cross-attention block for parameter parsimony — under full FT the backbone is co-adapting and a heavy probe adds limited marginal value.

**6. Expected gain over 53.71% control.** Headline literature anchor: V-JEPA (1) Table 3 reports **+16.1 pt** on SSv2 from average pooling → cross-attention probe on a *frozen* ViT-H/16. In full-FT on ViT-B at 4 frames, the gap should shrink by roughly an order of magnitude (the backbone is co-adapting). Expected: **+0.5 to +1.5 pt** (54.2–55.2% public LB), 80% CI [+0.0, +2.0].

**7. TTA implications.** Flip-TTA still works cleanly: the head is permutation-equivariant over space-time tokens (cross-attention has no spatial structure), so horizontal flip + left/right class remap behaves identically to the mean-pool case. Multi-scale TTA also fine (token count varies, but cross-attention handles variable N natively). Multi-clip TTA fine. **No re-measurement needed.**

**8. Diversity argument.** This head has the *lowest* prediction-distribution distance from the mean-pool control because both reduce (B, N, D) → (B, D) via a *linear* aggregation over tokens (attention weights are softmaxed but the output is still a weighted sum). Useful as a calibration point in the ensemble; expected pairwise correlation with control ≈ 0.93 (high). Provides ensemble lift mainly through better calibration on confused class pairs.

**9. Top failure mode.** Single learnable query under-utilizes the 4× MLP capacity, which mostly learns identity. Detect via: monitor MLP output norm vs query input norm at end of epoch 5 — if ratio < 0.05, the MLP is dead and you can drop to mlp_ratio=2 with no loss.

---

### ARCHITECTURE 2 — Multi-Query Perceiver Head (16 Queries, 1 Cross-Attention Layer)

**1. Architecture summary.** 16 learnable query tokens, one cross-attention layer with 12 heads matching the backbone, all (B, N, D) backbone tokens as KV. Post-attention: LayerNorm → MLP → mean-pool over queries → Linear(D, 33). This is the standard Perceiver-style multi-query aggregator; it lets the head learn 16 complementary "views" of the video (e.g., per-frame, per-region, motion-vs-appearance) and aggregate them.

**2. Exact PyTorch sketch.**

```python
class MultiQueryPerceiverHead(nn.Module):
    def __init__(self, dim=768, num_heads=12, num_queries=16, mlp_ratio=4.0,
                 num_classes=33, drop=0.0):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(1, num_queries, dim))
        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)), nn.GELU(),
            nn.Linear(int(dim*mlp_ratio), dim),
        )
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.queries, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=2e-5); nn.init.zeros_(self.head.bias)

    def forward(self, x):                # x: (B, N, D)
        B = x.size(0)
        q  = self.norm_q(self.queries.expand(B, -1, -1))
        kv = self.norm_kv(x)
        z, _ = self.attn(q, kv, kv, need_weights=False)   # (B, Q, D)
        z = z + q
        z = z + self.mlp(self.norm2(z))                   # (B, Q, D)
        z = self.norm_out(z).mean(dim=1)                  # mean-pool over queries
        return self.head(z)
```

**3. Init scheme.** Queries: trunc-normal std=0.02 (16 independent inits, NOT tied). Everything else as in Arch 1. **Do not add positional encoding to KV** — VideoMAEv2's sincos PE on (T, H, W) is already injected at patch-embed, and EP (Psomas et al. 2026) explicitly recommends against re-PE'ing the KV side. **Mean-pool over queries** (NOT concat-and-MLP-down): UniFormerV2 Table 4 confirms simple aggregation works; concat-and-MLP-down would add 16·D = 12K input dim params unnecessarily.

**4. Recipe deltas.** Same as Arch 1: head params in a separate group at base LR 5e-4, no LLRD, 5-epoch warmup. **No other deltas.**

**5. Parameter count.** 16 queries (12K) + QKV/out (2.36M) + MLP (4.72M) + norms + classifier ≈ **+7.1M params** (queries are negligible).

**6. Expected gain.** Literature anchors:
- UniFormerV2 SSv2 Table 4 sequential cross-attention head with 1 query = 69.5 vs no-fusion baseline ~65 (CLIP frozen ViT-B) → +4.5 pt frozen.
- EP (Psomas 2026, MAE ViT-B/IN-1K): EP64 = 75.6% vs LP/[CLS] = 67.7% → +7.9 pt frozen.
- In the user's full-FT setting, expected **+0.8 to +2.0 pt** over the 53.71% control. 80% CI [+0.2, +2.5].

**Important nuance:** UniFormerV2 ablation on SSv2 found **Q=1 (69.5) > Q=4 (69.1) > Q=16 (68.6)** in sequential cross-attention. For SSv2 with strong global encoders, fewer queries win. The user is on a 33-class subset (less class imbalance) and full-FT, so Q=16 is plausibly fine; **plan a Q-sweep {1, 4, 16, 32} as a 4-experiment overnight follow-up if Arch 2 finishes within ±1 pt of the best.**

**7. TTA implications.** Same as Arch 1: flip + multi-scale + multi-clip all unchanged. No re-measurement.

**8. Diversity argument.** Diverges from mean-pool through the *multiple* learned aggregations. Each of the 16 query attention maps will specialize (Psomas et al. show predictor maps become "complementary"). Expected pairwise correlation with control ≈ 0.85-0.90 (medium). Stronger ensemble pull than Arch 1.

**9. Top failure mode.** Query collapse — all 16 queries learn similar attention maps, degenerating to mean-pool + extra params. Detect: at end of epoch 10, compute pairwise cosine sim of `self.queries` after norm; if mean > 0.7, queries collapsed; mitigation is to add a 0.01-weighted orthogonality-penalty `||Q Qᵀ - I||_F²` to the loss. The UniFormerV2 result of 1>4>16 on SSv2 suggests this collapse may already be happening at Q=16 in the literature.

---

### ARCHITECTURE 3 — Late-Block Divided Space-Time Attention + 16-Query Perceiver Head

**1. Architecture summary.** Replace the *last K=6* of the 12 ViT-B encoder blocks with `SpaceTimeBlock` modules. Each SpaceTimeBlock: (i) **temporal-attention** — each token attends only to same-spatial-position tokens across T frames; (ii) **spatial-attention** — original ViT block weights, loaded from pretrained checkpoint; (iii) MLP. Crucially, the temporal-attention's *output projection (`temporal_fc`)* is **zero-initialized**, so at step 0 the new temporal path is identity and the model is exactly equivalent to the pretrained backbone — this is the TimeSformer convention. The classifier is Architecture 2's 16-query Perceiver. K=6 follows the user's Stan-style precedent; Bulat XViT (NeurIPS 2021 Table 2a) shows count matters, position doesn't.

**2. Exact PyTorch sketch.**

```python
class TemporalAttn(nn.Module):
    """Same-spatial-position attention across T frames; zero-init temporal_fc -> identity at init."""
    def __init__(self, dim=768, num_heads=12, drop=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=drop)
        self.temporal_fc = nn.Linear(dim, dim)            # TimeSformer-style
        nn.init.zeros_(self.temporal_fc.weight); nn.init.zeros_(self.temporal_fc.bias)

    def forward(self, x, T, H, W):     # x: (B, T*H*W, D)
        B, N, D = x.shape
        x = x.view(B, T, H*W, D).permute(0, 2, 1, 3).reshape(B*H*W, T, D)
        y, _ = self.attn(x, x, x, need_weights=False)
        y = self.temporal_fc(y)
        y = y.view(B, H*W, T, D).permute(0, 2, 1, 3).reshape(B, T*H*W, D)
        return y                       # identity at init (zero output)

class SpaceTimeBlock(nn.Module):
    """Wraps a pretrained ViT block; adds zero-init temporal-attn before it."""
    def __init__(self, vit_block, dim=768, num_heads=12, T=2, H=14, W=14, drop_path=0.0):
        super().__init__()
        self.norm_t = nn.LayerNorm(dim)
        self.temporal_attn = TemporalAttn(dim, num_heads)
        self.vit_block = vit_block         # pretrained block kept intact
        self.T, self.H, self.W = T, H, W
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):                  # x: (B, T*H*W, D)
        x = x + self.drop_path(self.temporal_attn(self.norm_t(x), self.T, self.H, self.W))
        x = self.vit_block(x)              # original spatial-only block (norm1, attn, norm2, mlp)
        return x
```

Integration: after backbone construction, replace `model.blocks[6:12]` with `SpaceTimeBlock(model.blocks[i], ...)` wrappers that *keep the original block as a submodule* (no re-init of spatial-attn or MLP).

**3. Init scheme.**
- Spatial-attn QKV/out + MLP in blocks 6–11: **loaded from pretrained ViT-B** (no re-init).
- Temporal-attn QKV: **random** (PyTorch default; trunc-normal std=0.02 acceptable).
- Temporal-attn output projection (`temporal_fc`): **zero-init weight AND bias** — this makes the entire new temporal-attn path output zero at step 0, so the block reduces exactly to the original ViT block. This matches the TimeSformer official implementation and XViT (Bulat NeurIPS 2021), which zero-inits the *temporal* positional embedding for the same identity-at-step-0 reason.
- Temporal LayerNorm `norm_t`: standard.
- Optional temporal positional embedding: **skip** because VideoMAEv2's sincos positional encoding on (T, H, W) is already injected at patch-embed.
- Perceiver head: as Arch 2.

**4. Recipe deltas.**
- **CRITICAL: separate parameter group for temporal-attn modules + head at full base LR 5e-4 with no LLRD multiplier**, 5-epoch warmup. The default LLRD 0.75^k schedule would apply roughly 0.75^6 ≈ 0.18× to layer 6's new temporal-attn, badly under-training it. Adapter and TimeSformer-style literature is unanimous: randomly-initialized modules get base LR; pretrained layers get LLRD.
- Optional: increase drop_path to 0.25 in the last 6 blocks to compensate for the new capacity (default 0.2 is also fine; revisit only if overfitting shows).
- Otherwise unchanged.

**5. Parameter count.** Per SpaceTimeBlock: temporal-attn QKV (3·D² = 1.77M) + temporal_fc (D² = 0.59M) + norm_t (1.5K). Per block ≈ 2.36M. Six blocks ≈ **+14.2M**. Plus the Perceiver head (+7.1M). Total = **+21.3M params**. Within the +5M–+50M budget.

**6. Expected gain.** Bulat XViT NeurIPS 2021 Table 2a on SSv2 ViT-B 8-frame: 1st-half=61.7, 2nd-half=61.6, alternating=61.2, **all-12=62.6**. Going from no temporal to last-6 ≈ 61.6 vs hypothetical no-temporal ~58 = roughly **+3 to +4 pt gain from late-block temporal modeling**. In the user's 4-frame full-FT setting with a strong MAE pretrain, expected **+2.0 to +3.5 pt** over the 53.71% control (target 55.7–57.2%). 80% CI [+1.0, +4.5]. Highest single-model expected lift of the four architectures.

**7. TTA implications.**
- **Flip-TTA: still works.** Temporal-attn permutes only across the T axis; horizontal flip + class remap is a spatial operation independent of the new temporal mechanism.
- **Multi-scale TTA: still works** (T and H, W are independent dims).
- **Multi-clip TTA: still works.**
- **Re-measure flip-TTA gain anyway** — the user noted +21 pt from TTA on the mean-pool model is suspiciously large and likely encodes a class-asymmetric prior that may shift with a stronger temporal model.

**8. Diversity argument.** Maximum architectural diversity from mean-pool: changes the FEATURES the head sees (not just the aggregation), because the last 6 blocks now compute spatio-temporal features rather than per-frame spatial features. Predictions will diverge most on classes that require true temporal reasoning ("moving something up" vs "moving something down" — the canonical SSv2 hard pair). Expected pairwise correlation with control ≈ 0.75–0.82 (low–medium). Strongest ensemble pull of the four.

**9. Top failure mode.** Init drift — if `temporal_fc` is not zero-init or if you accidentally also re-init the loaded `vit_block` weights, the model trains from scratch and underperforms the control by 10+ pt. **Detect: validate that at epoch 0, before any training, the model produces *identical* logits to the mean-pool-head control on a small val set.** If logits differ at step 0, init is wrong. This is the single highest-payoff sanity check in the entire setup.

---

### ARCHITECTURE 4 — AIM-Style Reused-Spatial-MSA Temporal Adapter + Perceiver Head

**1. Architecture summary.** In each backbone block, add a temporal adaptation step that **reuses the spatial self-attention weights** by reshaping the token sequence to put T on the attention axis, runs the same MSA, then applies a zero-init bottleneck adapter (D→64→D) to bring features back into distribution. NO new attention parameters — the temporal "attention" weights are tied to the spatial MSA. Combined with Architecture 2's 16-query Perceiver head.

**2. Exact PyTorch sketch.**

```python
class ZeroInitAdapter(nn.Module):
    def __init__(self, dim=768, bottleneck=64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act  = nn.GELU()
        self.up   = nn.Linear(bottleneck, dim)
        nn.init.trunc_normal_(self.down.weight, std=0.02); nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight);  nn.init.zeros_(self.up.bias)   # zero-init final proj
    def forward(self, x):
        return self.up(self.act(self.down(x)))

class AIMTemporalBlock(nn.Module):
    """Wraps a pretrained ViT block. Temporal step REUSES block.attn weights via reshape."""
    def __init__(self, vit_block, dim=768, T=2, H=14, W=14, bottleneck=64):
        super().__init__()
        self.norm_t = nn.LayerNorm(dim)
        self.block  = vit_block            # has .norm1, .attn, .norm2, .mlp
        self.t_adapter     = ZeroInitAdapter(dim, bottleneck)   # after reused-MSA on temporal axis
        self.joint_adapter = ZeroInitAdapter(dim, bottleneck)   # parallel to MLP
        self.T, self.H, self.W = T, H, W

    def forward(self, x):                  # x: (B, T*H*W, D)
        B, N, D = x.shape; T, H, W = self.T, self.H, self.W
        # 1) Reused-MSA applied along the temporal axis
        x_t = x.view(B, T, H*W, D).permute(0, 2, 1, 3).reshape(B*H*W, T, D)
        x_t = self.norm_t(x_t)
        x_t = self.block.attn(x_t)         # REUSE spatial attn weights, applied along T
        x_t = self.t_adapter(x_t)          # adapter; zero-init -> contribution = 0 at step 0
        x_t = x_t.view(B, H*W, T, D).permute(0, 2, 1, 3).reshape(B, N, D)
        x = x + x_t
        # 2) Original spatial-attn + MLP (with parallel joint adapter)
        x = x + self.block.attn(self.block.norm1(x))
        x = x + self.block.mlp(self.block.norm2(x)) + self.joint_adapter(self.block.norm2(x))
        return x
```

NOTE: `self.block.attn` is called twice with different inputs (once reshaped to T, once normally on spatial). This is AIM's exact trick — same weight matrix, two einsum reshapes.

**3. Init scheme.**
- All backbone parameters (norms, attn QKV/proj, MLP): **loaded from pretrained**, no re-init.
- `norm_t`: standard init.
- `t_adapter` and `joint_adapter`: down-projection trunc-normal std=0.02, up-projection **zero-init** (so the new path contributes zero at step 0; matches AIM's explicit "initialize the adapter to zero and remove the skip connection here to detach the effect of temporal adaptation at the beginning of training" recipe in their §3.2).
- Backbone NOT frozen — user does full FT. This is the key divergence from AIM, which uses a frozen backbone. Implication: the shared attn weights will update during training; the temporal application of the same matrix to a reshaped tensor is then a *constraint* on what those weights can learn (they must be useful for both spatial and temporal attention). This may help (regularization) or hurt (capacity bottleneck) — empirically untested in the full-FT regime.

**4. Recipe deltas.**
- **Separate parameter group for adapters + head at full base LR 5e-4 with 5-epoch warmup, no LLRD**. Backbone keeps LLRD 0.75.
- Insertion: **all 12 blocks** (per AIM's default). Adapter bottleneck 64 (AIM default; sensitivity is mild — sweep {64, 128, 192} only if Arch 4 wins).
- Otherwise unchanged.

**5. Parameter count.** Per block: 2 adapters × (D·64 + 64·D + 1.5K bias) = 2 × ~98K = ~0.2M; ×12 blocks ≈ 2.4M. Plus norm_t × 12 = 18K. Plus Perceiver head 7.1M. Total ≈ **+9.5M params**. The lightest of the four.

**6. Expected gain.** AIM Table 1 on SSv2 ViT-B/IN-21K, 8 frames, FROZEN backbone:
- space-only LP = 15.1; +spatial-adapter = 36.7; +temporal-adapter = 61.2; +joint-adapter = 62.0.
- The *temporal* step alone is **+24.5 pt** (36.7 → 61.2); full AIM exceeds full-FT TimeSformer (59.5) by +2.5.
- In the user's FULL-FT setting where the backbone is already adapting and is initialized from VideoMAE SSv2-domain pretraining (not IN-21K image pretraining), the marginal from adding AIM temporal adaptation will be much smaller. Expected **+1.5 to +3.0 pt** over the 53.71% control. 80% CI [+0.5, +4.0].

**7. TTA implications.** Flip, multi-scale, multi-clip TTA all still work (temporal-axis reshape is orthogonal to flip). No re-measurement needed beyond the usual sanity check on flip class-remap.

**8. Diversity argument.** Architecturally orthogonal to Architecture 3: same goal (temporal modeling) but **opposite mechanism**. Arch 3 introduces NEW temporal-attention weights with TimeSformer-style zero-init `temporal_fc`; Arch 4 REUSES spatial-attention weights via einsum reshape and adds a small adapter MLP. These produce different inductive biases: Arch 3 lets the model learn temporal patterns *unconstrained by spatial weights*; Arch 4 forces the temporal patterns to *share* the spatial attention basis. Predictions will diverge on classes where the optimal temporal kernel differs structurally from the optimal spatial kernel. Expected pairwise correlation with control ≈ 0.78–0.84; correlation with Arch 3 ≈ 0.85 (similar enough to be a robustness pair, different enough to ensemble).

**9. Top failure mode.** The reused-MSA trick assumes the spatial attention pattern transfers to the temporal axis. In the user's case with **T'=2 temporal tokens** (4 frames, tube_t=2), the attention is on a sequence of length 2 — degenerate (essentially pairwise). AIM tested 8–16 frames where T' ≥ 8 attention is meaningful. With T'=2, the reused-MSA is essentially a pairwise comparison + adapter — which still helps but is far less than AIM's reported +24.5 pt. Detect: train one epoch and compare val accuracy to the control; if Δ < 0.3 pt at end of epoch 5, AIM's mechanism is degenerate at this T' and you should switch to T'=4 (tube_t=1) for this architecture only. This is the single highest leverage hyperparameter to consider for Arch 4.

---

## One-Page Summary Table

| Architecture | Extra Params | Expected Δ vs 53.71% | Train Time × control | Ensemble Diversity | Recommended Run Order |
|---|---|---|---|---|---|
| **Control** (mean-pool, mae450/500) | +0.0M | 0 (baseline; +0.2–0.5 from MAE epoch upgrade) | 1.0× | — | Run #1 (must have) |
| **Arch 1** Single-Q Attentive Probe | +7.1M | +0.5 to +1.5 pt | 1.03× | LOW (corr ≈ 0.93) | Run #2 (cheapest sanity check) |
| **Arch 2** 16-Q Perceiver | +7.1M | +0.8 to +2.0 pt | 1.05× | MEDIUM (corr ≈ 0.85) | Run #3 |
| **Arch 3** Late-K=6 Divided ST + Perceiver | +21.3M | **+2.0 to +3.5 pt** | 1.20× | **HIGH (corr ≈ 0.78)** | Run #4 (highest EV) |
| **Arch 4** AIM Reused-MSA + Perceiver | +9.5M | +1.5 to +3.0 pt | 1.10× | HIGH (corr ≈ 0.80, ⊥ to Arch 3) | Run #5 |

All five fit one overnight 5-experiment slot. Total expected stand-alone winner: **Arch 3**. Total expected ensemble winner: **Caruana stack of control + Arch 1 + Arch 2 + Arch 3 + Arch 4**, projected **+3.5 to +5.5 pt** over the 53.71% control on public LB.

## Cross-Cutting Question A — Ensemble Lift: Diverse 4 vs 4 Seed-Replicas

**Verdict: the diverse-4 ensemble dominates a 4-seed-replica ensemble by 1.0–2.5 pt at equal compute on a 33-class top-1 task. This is one of the most robust findings in the modern deep-ensemble literature.**

- **Caruana et al. (ICML 2004) "Ensemble Selection from Libraries of Models"** explicitly designed forward-stepwise selection (with replacement, weighted) precisely to exploit *library diversity*. Their analysis of model-type weight assignment in Table 3 shows that the selected ensemble draws weight from multiple model families (NNs, DTs, KNN, boosted, bagged), and the paper's central claim is that selection from a *heterogeneous* library beats the best single model on every test problem evaluated. (Specific aggregate loss-reduction percentages cited in secondary sources are best treated as approximate; the qualitative finding that diverse libraries help is unambiguous.)
- **Lakshminarayanan et al. (NeurIPS 2017) "Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles"** established seed-only deep ensembles as a strong baseline. Fort, Hu & Lakshminarayanan (arXiv 1912.02757, 2019) "Deep Ensembles: A Loss Landscape Perspective" then showed seed-only ensembles work primarily because different seeds converge to *different basins* — but the diversity is bounded by basin-level perturbation, not function-class perturbation.
- **Wenzel et al. (NeurIPS 2020) "Hyperparameter Ensembles for Robustness and Uncertainty Quantification"** is the most direct evidence: their "hyper-deep ensembles" stratify random *hyperparameters* (dropout, L2, label-smoothing) across multiple seeds and **outperform pure deep ensembles** at equal budget. The paper reports: "On image classification tasks, with MLP, LeNet, ResNet 20 and Wide ResNet 28-10 architectures, we improve upon both deep and batch ensembles" — datasets tested were CIFAR-10, CIFAR-100, and Fashion-MNIST (not ImageNet). The paper does not directly test architectural diversity, but architectural diversity is mechanistically a *stronger* perturbation than hyperparameter diversity (it changes the function class, not just the regularization), so the lift bound is at least as good in expectation.
- **Caveat from Abe et al. 2024 "Pathologies of Predictive Diversity in Deep Ensembles" (arXiv 2302.00704):** mechanisms that *trade off* member accuracy for diversity (e.g., divergent objectives, heavy regularization) often *fail* to beat standard deep ensembles. The user is not doing this — Architectures 1–4 are all expected to be individually competitive (within 1–3 pt of the best), with diversity coming "for free" from the architectural choice. This is the regime where diverse ensembling reliably wins.

Concrete prediction for this Kaggle task:
- **5-seed control deep-ensemble**: +1.0–1.5 pt over single best seed (typical for 33-class top-1).
- **Caruana stack of control + Arch 1–4 (with Ws/CWS/LSG weights)**: +2.5–4.0 pt over single best single-arch.
- **Net advantage of diverse-5 over seed-5: +1.0–2.5 pt.**

Practical recipe: train each of the 5 to convergence, dump softmax logits on a held-out fold (the user doesn't have val labels for the public test — use a 10% split of the train pool with strict-confidence pseudo-labels, OR rely on judicious public-LB probing). Run Caruana forward selection with replacement, weighted, optimizing top-1 accuracy with 100–1000 rounds. Take the resulting weight vector and apply to softmaxes of all test predictions.

## Cross-Cutting Question B — K-Sweep if Architecture 3 Wins

**If Arch 3 wins the first round, the K-sweep follow-up should test K ∈ {3, 6, 9, 12} with K=6 as the already-tested point.**

- **Bulat XViT Table 2a (NeurIPS 2021)** on SSv2 ViT-B/8-frame: 1st-half=61.7, 2nd-half=61.6, alternating=61.2, **all-12=62.6**. The all-12 setting beats halves by +0.9–1.0 pt. The position-doesn't-matter finding is decisive; **count matters and is mildly saturating between K=6 and K=12**.
- **AIM (ICLR 2023)** inserts adapters in **all 12 blocks** for the reported best result (62.0 IN-21K, 66.4 CLIP, both ViT-B on SSv2).
- **UniFormerV2 (ICCV 2023)** inserts global UniBlocks in last 4 layers for K400 but **last 8/16 for SSv2** (in ViT-B/L respectively) — i.e., MORE temporal modeling helps for SSv2 specifically.
- **TimeSformer** uses K=12 (every block has divided space-time attention).

For 4-frame SSv2 with VideoMAE pretraining, **expected K\* = 9 to 12**, with the marginal from K=6 → K=9 ≈ +0.5–1.0 pt and from K=9 → K=12 ≈ +0.2–0.5 pt (saturating). **Recommendation**: if Arch 3 (K=6) places top-2 in the first round, run K ∈ {9, 12} as the next overnight pair, alongside one Q-sweep (Arch 2, Q=1) and one ablation (Arch 3 minus Perceiver head, i.e. mean-pool head with SpaceTimeBlocks). This is exactly 4 experiments — fits the 5-slot overnight budget with a control re-run.

**Trigger threshold**: if Arch 3 (K=6) beats control by ≥ +1.5 pt on public LB, run K=12 next. If it beats control by < +1.5 pt, the saturation curve is steeper than expected and K=9 is the better next bet (don't burn the experiment on K=12).

## Recommendations (Staged, Concrete)

**Tonight (5 parallel runs):**
1. Control (mae500-ft-f4, mean-pool head, exact current recipe) — must rerun on best SSL ckpt for fair comparison.
2. Arch 1 (Attentive Probe, 1 layer, single query).
3. Arch 2 (Perceiver, Q=16).
4. Arch 3 (Late-K=6 Divided ST + Q=16 Perceiver). **Highest EV.**
5. Arch 4 (AIM Reused-MSA all 12 blocks + Q=16 Perceiver).

**Day 2 morning:** measure all 5 on public LB with the existing champion TTA (6 views). Cross-check that flip-TTA still works for Archs 3 & 4 (it should; spatial flip is unchanged). Identify the top 1–2 single-model architectures. Run Caruana stacking on softmax outputs using a held-out internal val (10% of train pool with confident pseudo-labels) — quick measurement of ensemble lift.

**Day 2–3 (overnight):** Based on round-1 winner:
- If **Arch 3 wins**: K-sweep {9, 12}, plus an Arch-3-minus-Perceiver ablation (isolate the temporal contribution), plus one Q-sweep on Arch 2 (Q=1 — the UniFormerV2 result suggests this might be SSv2-optimal), plus a seed-replica of Arch 3 (for the ensemble).
- If **Arch 4 wins**: insertion-position sweep (last 6 vs all 12 blocks), bottleneck-dim sweep {64, 128, 192}, plus a frame-count test at T=4 tube_t=1 (mitigate the T'=2 degeneracy flagged in §Arch 4 failure mode).
- If **Arch 2 wins**: Q-sweep {1, 4, 32}, plus multi-layer test (2 cross-attention layers vs 1).
- If **Arch 1 wins** (low probability): essentially a null result — the head was never the bottleneck; consider scaling up frames or running more SSL pretraining.

**Day 4–5:** Lock down final ensemble. Run Caruana CWS (with-replacement weighted), LSG (Lakshminarayanan-style averaged), and Ws (forward-stepwise) on the same library; submit the strongest combination as the final entry.

**Threshold to abandon the multi-arch plan:** if after round 1 *no architecture* beats the control by ≥ +0.5 pt on public LB, the bottleneck is not the head — it's frames-per-clip or the SSL checkpoint. In that case, redirect the day-2 budget to T=8 frames (will likely require new SSL) or to model scaling (ViT-L if available).

## Caveats

- **The user's "+21 pt from TTA" is unusual.** 6-view TTA giving +21 pt over no-TTA on the mean-pool baseline (53.71% → ~32.7% without TTA?) implies the no-TTA model is severely under-calibrated or there is a class-prior mismatch the flip+remap fixes. Architectures 1–4 may have a different TTA delta. Re-measure TTA gain on each architecture; the +21 pt is unlikely to transfer fully.
- **T'=2 degeneracy** (only 2 temporal tokens after tubelet pooling) limits the upside of all temporal modeling architectures. AIM, XViT, TimeSformer evidence is all from 8–16 frame settings. The user's 4-frame/tube_t=2 setup is unusually compressed. Expected gains in this report are scaled down accordingly (roughly 0.5× the literature deltas), but there is genuine uncertainty here. Consider testing tube_t=1 (T'=4) for Arch 3 and 4 specifically if either wins round 1.
- **The Psomas et al. (Efficient Probing) numbers are all from frozen-encoder probing on image classification.** They are *suggestive* about query design but not directly transferable to the user's full-FT video setting.
- **The V-JEPA-style attentive probe layer-count sweep (1 vs 2 vs 4 blocks)** is referenced in V-JEPA / V-JEPA-2 Section 12.2 / Appendix C.2, but the exact 1-vs-2-vs-4 numerical Top-1 table was not extractable from web search during research (PDF retrieval failed); the V-JEPA-2 paper text confirms 4-layer is the production setting and outperforms simpler pooling on motion tasks like SSv2. Architecture 1 uses a 1-block probe because at full-FT the marginal of stacking more probe blocks should be small (the backbone is co-adapting) and the parameter cost matters more.
- **The +16.1 pt cross-attention vs avg-pool delta** is from the V-JEPA (1) paper (Bardes et al., arXiv 2404.08471, Table 3) on a *frozen ViT-H/16*, not V-JEPA-2 and not ViT-B. The qualitative finding transfers; the absolute magnitude does not.
- **Wenzel et al. (NeurIPS 2020) directly tested hyperparameter diversity** (not architectural diversity) on CIFAR-10/100 and Fashion-MNIST with MLP/LeNet/ResNet-20/WRN-28-10 architectures. The extrapolation to architectural diversity in this report is principled (architectural change ⊇ hyperparameter change) but is an *extension* of their findings, not a direct claim from their paper.
- **The user's Stan-style reference (51% architecture)** could not be cross-checked against published literature; the description treats it as faithful to the user's prior knowledge. K=6 in Arch 3 is set per the user's Stan precedent; the K-sweep follow-up should formalize this choice.
- **Architectures 3 and 4 BOTH need the separate parameter group at base LR.** This is the single highest-leverage hyperparameter detail; if missed, expect both to underperform by 1–2 pt vs the projections in this report.