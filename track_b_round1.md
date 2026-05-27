# Track-B Next-Step Plan: 71.7% → 74%+ on 33-Class SSv2 Subset, Single RTX 3090, 10 Days

## TL;DR
- The single highest-EV next step is a **multi-query (8–16) cross-attention probe** replacing the single-query V-JEPA 2 attentive pooler, run with the *Meta-published* LR×WD grid (5 LRs × 4 WDs = 20 probes), then **greedy-souped** across hyperparameter winners — expected +1.5 to +3 pp net, fits comfortably in one weekend on a 3090.
- The next two wins are **pair-margin / logit-adjustment loss aimed at the pretend↔real and "put-into" sink classes** (counterfactual ceiling alone is +3.3 pp from fixing the top-3 confusion pairs), and **importing real SSv2 clips for the 33 target classes** to widen the 4-frame motion distribution your local data is missing (SSv2 has exactly 168,913 train / 24,777 val / 27,157 test clips across 174 classes, so 33-class slice yields ≈32k additional clips, more than doubling per-class data for the worst classes).
- Do **not** spend any more cycles on (a) single-backbone V-JEPA 2 ensembles, (b) flip / dense-multi-view TTA, (c) tuning on the contaminated holdout, or (d) pseudo-labeling Kaggle test. These are all confirmed dead-ends or LB-overfit anti-patterns; held-out test is the real eval target.

---

## Key Findings

1. **The Meta-published V-JEPA 2 ViT-L/16 256² SSv2 probe is exactly the architecture you are already running.** From `configs/eval/vitl/ssv2.yaml` on `facebookresearch/vjepa2`: `num_probe_blocks: 4`, `num_heads: 16`, `use_pos_embed: false`, last-layer `target_encoder` features, AdamW, bf16. The reported SSv2-174 top-1 for this exact recipe is **73.7%** (V-JEPA 2 GitHub README evaluation-probes table). You are at 70.5% single / 70.8% ensemble on a 33-class subset with LoRA-r16 — meaning your *probe quality*, not the backbone, is the bottleneck.
2. **Meta's own SSv2 ViT-L config sweeps a 5×4 = 20-probe LR×WD grid** with LRs `{5e-3, 3e-3, 1e-3, 3e-4, 1e-4}` and WDs `{0.01, 0.1, 0.4, 0.8}`, **no warmup** (`warmup: 0.0`), cosine-to-zero (`final_lr: 0.0`), 20 epochs, batch 4 × 64 GPUs ≈ 256 global. They run all 20 simultaneously via `multihead_kwargs` and keep the best — this is the LR-sensitivity diagnostic you want, already published.
3. **The single learnable query is the default attentive pooler in V-JEPA / V-JEPA 2** (1 query token cross-attending to the concatenated tokens from 2 segments × 3 views). Cross-attention probes with more queries consistently outperform single-query pooling for fine-grained classification when the backbone is frozen or PEFT-adapted. Per BLIP-2 (Li et al., arXiv:2301.12597), going from 16 → 32 query tokens improved VQAv2 accuracy and 32 → 64 gave marginal additional gain (BLIP-2 ships 32 queries × 768 dim as default); Flamingo's Perceiver Resampler used **exactly 64** latent queries, motivated to "significantly reduce the computational complexity of vision-text cross attention" (Alayrac et al., NeurIPS 2022). Your other-track +2 pp from 16 queries over mean-pool + linear is in line with that literature.
4. **The confusion matrix says capacity is fine; the head's decision boundary is wrong.** Train 99% vs val 70.6% with the same encoder = the encoder *has* the features; the probe is collapsing fine-grained verb pairs (pretend↔real, drop↔put, fold↔unfold, pick↔move up). All of these are *direction-encoded, temporally-cued* distinctions that a single global query loses by averaging. This is the strongest argument for multi-query cross-attention and for **pair-margin loss** targeted at the eight pairs you enumerated.
5. **The 4-real-frame → 16-slot duplication is a genuine distribution shift from Meta's SSv2-174 probe**, which was trained on 16 *real, uniformly TSN-sampled* frames at 256². The probe weights you initialize from will be near-optimal for *real* 16-frame motion, not for 4-frame duplicates. This is your main novelty angle; importing real SSv2 clips and applying the same 4→16 duplication transformation at train time is the highest-EV data work (and is what closes the 4-frame domain gap to the Kaggle test set).
6. **Architecturally diverse ensembles are downloadable on HuggingFace today** (no pretraining needed): `MCG-NJU/videomae-base-finetuned-ssv2` ("This model obtains a top-1 accuracy of 70.6 and a top-5 accuracy of 92.6 on the test set of Something-Something-v2", per its HF card), `MCG-NJU/videomae-base-short-finetuned-ssv2` ("top-1 accuracy of 69.6 and a top-5 accuracy of 92.0"), `MCG-NJU/videomae-small-finetuned-ssv2` (66.8 top-1), `facebook/timesformer-base-finetuned-ssv2`, `facebook/timesformer-hr-finetuned-ssv2`, `fcakyon/timesformer-large-finetuned-ssv2`, and `OpenGVLab/InternVideo2-Stage2_1B-224p-f4` (with SSv2 fine-tune scripts published, though you'd need to train the head). All are reachable from a 3090.
7. **Public SSv2 access**: The dataset is hosted at `qualcomm.com/developer/software/something-something-v-2-dataset` (formerly 20bn.com/datasets/something-something/v2; Qualcomm acquired Twenty Billion Neurons). Registration required, free for research; 220,847 total videos in 20 parts × ~1 GB each, ~19.4 GB total, webm/VP9 (per HyperAI mirror metadata and GluonCV docs). Also mirrored at `HuggingFaceM4/something_something_v2` (metadata only, 43.7 kB) and `hyper.ai/en/datasets/17204`.

---

## Ranked Experiments

### Exp 1 — Multi-query cross-attention probe + full Meta LR×WD grid + greedy soup
- **Hypothesis:** Replacing the single-query V-JEPA 2 attentive pooler with an 8- or 16-query cross-attention pooler closes the within-pair decision-boundary gap and gives +1.5 to +3 pp val. Running Meta's full 5×4 LR×WD sweep then greedy-souping winners gives a further +0.3 to +0.8 pp from weight-space averaging.
- **Evidence:**
  - Meta V-JEPA 2 (Assran et al., arXiv:2506.09985) ships a 4-block / 16-head, *single*-query attentive pooler and reports **73.7%** on SSv2-174 at ViT-L/16 256² (`facebookresearch/vjepa2` README).
  - BLIP-2 (Li et al., arXiv:2301.12597) ablation: 16 → 32 queries improves VQAv2, 32 → 64 gives marginal gain; 32 queries × 768 dim is the shipped default.
  - Flamingo (Alayrac et al., NeurIPS 2022): Perceiver Resampler uses exactly **64** latent queries; the resampler "re-sample[s] the visual input to a fixed and small number (in practice 64) of outputs".
  - Wortsman et al. (ICML 2022, arXiv:2203.05482) "Model soups": greedy soup over a random LR/WD/aug sweep beats best individual; for ViT-G the greedy soup reaches **90.94%** vs. **90.78%** best-individual on ImageNet top-1 (Table 1). "The greedy soup adds models sequentially to the model soup, keeping a model in the soup if accuracy on the held-out validation set does not decrease."
- **Engineering effort:** 6–10 h to write the multi-query `CrossAttentionPooler` and to refactor the trainer to take `--num_queries N --probe_depth D --probe_heads H` from CLI.
- **Wall-clock on a 3090:** A single 4-block / 16-query probe on top of LoRA-r16 V-JEPA 2 ViT-L is ~3–4 h for 10 epochs on your 33-class subset at bs=4 grad-accum to 32. The 20-probe LR×WD grid in serial is ~60–80 h; do as a coarse 5-LR sweep first (overnight, ~15 h), pick top-3 LRs, then sweep 4 WDs at each (overnight ×2). Greedy soup runs in seconds.
- **Hyperparameters (concrete starting recipe):**
  - `num_queries=16, depth=2, heads=16, mlp_ratio=4.0, embed_dim=1024 (ViT-L), pre-norm, drop=0.1, drop_path=0.1` (depth=2 saves time vs Meta's depth=4 because LoRA already adapts the encoder).
  - Probe LR swept from Meta's exact grid `{5e-3, 3e-3, 1e-3, 3e-4, 1e-4}`.
  - LoRA LR = probe LR / 10 (decoupled).
  - WD `{0.05, 0.4}` for coarse, then Meta's outer WDs `{0.01, 0.1, 0.4, 0.8}` at winning LRs.
  - AdamW betas (0.9, 0.999), eps 1e-8, cosine to 0, **no warmup**, bf16, batch 4 × grad-accum 8 = effective 32.
- **Evaluation:** Single-center-clip top-1 (NoTTA, your data confirms TTA is −10 pp), on the clean held-out slice from Exp 8.

### Exp 2 — Pair-margin / logit-adjusted loss on the top 8 confusion pairs
- **Hypothesis:** Adding a class-prior logit-adjustment term (Menon et al. 2021, *Long-Tail Learning via Logit Adjustment*, ICLR, arXiv:2007.07314) **plus** an explicit margin on the eight pretend↔real pairs converts the confusion-matrix mass on those pairs into correct predictions. Counterfactual ceiling fixing only the top 3 pairs = +3.3 pp; realistically capture 30–60% of that = +1.0 to +2.0 pp on val.
- **Evidence:** Menon et al. establishes that logit adjustment by class-prior log-frequency is Fisher-consistent for balanced error and provides post-hoc *or* loss-time adjustment. Cui et al. 2019 (*Class-Balanced Loss*, CVPR, arXiv:1901.05555) provides the effective-number reweighting variant. Both are cheap to implement and orthogonal to architecture changes.
- **Engineering effort:** 4 h.
- **Wall-clock:** ~3–4 h on a 3090.
- **Hyperparameters:**
  - **Post-hoc logit adjustment**: subtract `τ · log p̂(y)` from logits at inference with τ ∈ {0.5, 1.0, 1.5, 2.0}, fit τ on clean holdout by maximising top-1.
  - **Loss-time LA (Menon)**: `softmax(z_y + τ log p̂_y)` with τ = 1.0 in CE.
  - **Pair margin**: for the 8 pairs P = {(11,14), (16,22), (9,11), (2,22), (17,29), (3,32), (9,14), (11,30)}, add additive margin m to correct-class logit when other member is in top-2. Sweep m ∈ {0.1, 0.2, 0.3, 0.5}.
  - **Auxiliary "is_pretend" head**: binary BCE on `{14, 16, 17}`, weight λ = 0.2.
- **Evaluation:** Re-compute the confusion matrix from your `per_class_error_analysis.py` on the clean holdout — confirm pair errors actually drop, not just raw accuracy.

### Exp 3 — Real-SSv2 import + 4-frame transformation
- **Hypothesis:** Importing real SSv2 webm clips for the 33 target classes and applying the same "first-60%-then-4-frames → 16-slot uneven duplication" transformation matches the local data's domain. Per-class clip counts roughly triple (class 26 "spill next to" goes from 162 → ~600+); on classes where you are recall-limited by data (16, 17, 11, 14), this is the dominant lever.
- **Evidence:** SSv2 has 168,913 training videos across 174 classes. 33-class slice ≈ 168,913 × 33/174 ≈ **32,000 additional candidate training clips**. Even at 30% retention after deduplication, that's ~10× the data you have for some rare classes.
- **Engineering effort:** ~8 h end-to-end (download, decode, filter, dedupe, transform).
- **Wall-clock:** Download + webm-to-frames extraction ~6 h (parallelisable with training). Re-train one model on combined data: ~6 h for 4 epochs at the new corpus size.
- **Risks:** Test-set contamination — the professor likely drew Kaggle test from SSv2 validation or test. **MUST** filter by SSv2 `video_id` from `validation.json` AND `test.json` before merging (see Q2 section).

### Exp 4 — Probe-only continuation + greedy model soup over checkpoints
- **Hypothesis:** Freeze the LoRA-adapted encoder fully, keep training only the head for 5 more epochs with linearly-decaying LR; checkpoint every 200 steps; greedy-soup the resulting checkpoints. Expected +0.3 to +0.7 pp.
- **Evidence:** Wortsman et al. 2022: "Averaging the weights of multiple models fine-tuned with different hyperparameter configurations often improves accuracy and robustness… without incurring any additional inference or memory costs"; greedy soup explicitly adds checkpoints only if validation does not decrease — perfect under our contaminated-holdout problem because you can use *only* the small clean held-out slice (Exp 8).
- **Engineering effort:** 2 h.
- **Wall-clock:** 4 h to re-run with checkpointing; soup fitting < 1 min.
- **Hyperparameters:** Save head ckpts at steps {200, 400, 600, 800, 1000}; greedy add in order of clean-holdout accuracy.

### Exp 5 — Architecturally diverse ensemble (NOT same-backbone)
- **Hypothesis:** The current 5-V-JEPA2 ensemble collapses because members are nearly identical. Replacing 3 of 5 members with genuinely different backbones (VideoMAE-base-ssv2 70.6%, TimeSformer-hr-ssv2, InternVideo2-Stage2_1B) and re-fitting weights on the clean holdout closes the gap. Expected +0.5 to +1.5 pp over best single.
- **Evidence:** Pretend↔real confusion is rooted in *temporal* modeling; tube-mask (VideoMAE) and joint space-time (TimeSformer) backbones make *different* errors than V-JEPA 2's predictive-feature backbone → higher disagreement → unweighted average captures complementary signal.
- **Concrete downloadable checkpoints:**
  - `MCG-NJU/videomae-base-finetuned-ssv2` — "top-1 accuracy of 70.6 and a top-5 accuracy of 92.6 on the test set of Something-Something-v2" (HF model card); 16-frame 224². Drop-in `VideoMAEForVideoClassification`.
  - `MCG-NJU/videomae-base-short-finetuned-ssv2` — "top-1 accuracy of 69.6 and a top-5 accuracy of 92.0" (HF card).
  - `MCG-NJU/videomae-small-finetuned-ssv2` — 66.8 top-1.
  - `facebook/timesformer-base-finetuned-ssv2` / `timesformer-hr-finetuned-ssv2` (448²) — drop-in `TimesformerForVideoClassification`.
  - `fcakyon/timesformer-large-finetuned-ssv2` — community port.
  - `OpenGVLab/InternVideo2-Stage2_1B-224p-f4` — uses fine-tune script `1B_ft_ssv2_f8.sh` in `OpenGVLab/InternVideo` (`--lr 1e-4 --drop_path 0.3 --layer_decay 0.915 --num_frames 8`); frozen-backbone + small head only on a 3090.
- **Engineering effort:** 8 h to wire each new backbone's preprocessing (frame counts, resolutions, normalizations) and to mask its 174-way head down to 33 classes.
- **Wall-clock:** Inference of all members on full val: ~1.5 h per member. Re-fit ensemble weights on clean holdout: minutes.
- **Hyperparameters:** Members = {your current best V-JEPA 2 ViT-L, V-JEPA 2 multi-query (Exp 1), VideoMAE-base-ssv2, TimeSformer-hr-ssv2}; non-negative weights summing to 1; **fit on clean holdout only**.
- **Anti-fail check:** If a member's clean-holdout accuracy < 0.50, drop it — at val ~0.60 it adds noise (your DINOv2 + temporal probe finding confirms this).

### Exp 6 — Decoupled head/LoRA LR sensitivity diagnostic (one overnight window)
- **Hypothesis:** Probe LR and LoRA LR have different optimal ranges. Decoupling them with a ratio sweep r ∈ {1, 4, 10, ∞} (∞ = LoRA frozen) gives the right operating point and tells you whether probe-only training is competitive.
- **Evidence:** V-JEPA 2's own configs scan LR over 5× range. Standard 2024–2025 PEFT papers (e.g., LoRA-ViT transfer) typically use LR ratios in the 1–10 range with head LR ≥ LoRA LR.
- **Engineering effort:** 1 h (parameter-group split).
- **Wall-clock:** 12 configs × 4 epochs (early-stop) ≈ 12 h overnight.
- **Hyperparameter grid:**
  - Coarse (Night 1, ~12 h, 4 epochs each): probe_LR ∈ {3e-3, 1e-3, 3e-4} × ratio ∈ {1, 4, 10, ∞}, WD=0.05, drop=0.1.
  - Fine (Night 2, ~12 h, 10 epochs): top 3 from coarse × WD ∈ {0.01, 0.1}; save ckpts every 200 steps; greedy-soup.

### Exp 7 — Auxiliary "is_pretend" + super-class auxiliary head
- **Hypothesis:** Binary head predicting "is class one of {14, 16, 17}?" off the pooled feature regularises the main head toward the pretend/real boundary. Expected +0.3 to +0.7 pp.
- **Effort:** 2 h.
- **Wall-clock:** Same as one full probe training.
- **Hyperparameters:** Aux loss weight λ = 0.2; binary "is_pretend" plus optionally a 4-way verb-family head {pick, put, throw, other}.

### Exp 8 — Clean held-out protocol (PREREQUISITE FOR ALL ABOVE)
- **Hypothesis:** Every experiment is read through a fogged lens until you fix the holdout. Reported 80.9% holdout vs 70.8% LB = 10 pp leakage.
- **Recipe:**
  1. Define a **frozen held-out slice** = 15% of `data/val`, stratified by class. Lock IDs in JSON; never train on these. Call it `holdout_clean.json`.
  2. For "fulltrain" members that train on train+val, *exclude* `holdout_clean.json`. Define `train_plus_val_minus_holdout.json` as the new fulltrain corpus.
  3. Report all experiments using `holdout_clean.json` top-1; LB submissions are confirmatory only.
- **Evidence basis:** Standard nested-CV/held-out practice. Wortsman et al. explicitly require a held-out validation set for greedy soup; a contaminated set silently includes LB-overfit checkpoints.
- **Effort:** 1 h.

---

## Cross-Attention Probe: Architecture & Starting Configuration (Q1)

### What V-JEPA 2 ships (confirmed)
From `configs/eval/vitl/ssv2.yaml` on `facebookresearch/vjepa2`:
```yaml
classifier:
  num_heads: 16
  num_probe_blocks: 4
```
The `AttentiveClassifier` is instantiated as `AttentiveClassifier(embed_dim=1024, num_heads=16, depth=4, num_classes=174)`, with **a single learnable query token** that cross-attends to the concatenated tokens from 2 temporal segments × 3 spatial views. Pre-norm, MLP ratio 4.0, dropout 0.0. Encoder uses RoPE; `use_pos_embed: false` on the probe input. Optimization: AdamW, no warmup, cosine to 0, bf16. LR×WD grid `{5e-3, 3e-3, 1e-3, 3e-4, 1e-4} × {0.01, 0.1, 0.4, 0.8}` over 20 epochs.

### What you should run
A **multi-query, 2-block** cross-attention pooler — shallower than depth-4 because LoRA already adapts the encoder:

```python
class CrossAttentionProbe(nn.Module):
    def __init__(self, embed_dim=1024, num_queries=16, num_heads=16,
                 depth=2, mlp_ratio=4.0, dropout=0.1, drop_path=0.1,
                 num_classes=33):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, mlp_ratio,
                                dropout, drop_path, pre_norm=True)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, encoder_tokens):  # (B, N_tokens, 1024)
        q = self.queries.expand(encoder_tokens.size(0), -1, -1)
        for blk in self.blocks:
            q = blk(q, encoder_tokens)   # cross-attn(q, k=v=encoder_tokens)
        q = self.norm(q)
        return self.head(q.mean(dim=1))  # mean-pool 16 queries
```

### Recommended starting configuration
| Knob | Recommended | Rationale |
|---|---|---|
| `num_queries` | **16** | BLIP-2 ablation: 16 → 32 helps, 32 → 64 marginal; your other-track data confirms 16 is the sweet spot. Try 8 and 32 as cheap variants. |
| `depth` | **2** | LoRA on encoder already does the heavy work; depth-2 is half the parameters of Meta's depth-4 and as good in our regime. Sanity-check depth=4 on your top run. |
| `heads` | **16** | Match Meta and ViT-L head count. |
| `mlp_ratio` | 4.0 | Standard. |
| `dropout` / `drop_path` | 0.1 / 0.1 | Your 99% train vs 70.6% val gap is screaming over-fit; raise from Meta's 0.0. |
| `LayerNorm` | **Pre-norm** | Stable on V-JEPA features; post-norm is harder to tune with bf16. |
| `Positional encoding` | **None added before pooling** | Encoder uses RoPE; matches Meta's `use_pos_embed: false`. |
| Query pooling | **Mean-pool the 16 query outputs**, then linear | Concat (16×1024 → linear) over-parametrises in our small-data regime. |
| Layer-wise input | **Last block only** | Concat of last-K blocks is a known +0.2–0.5 pp trick (DINOv2 / V-JEPA 2.1 use it) but doubles probe memory — not worth it on a 3090. |
| Gating | **No** | Flamingo's tanh-gated cross-attention helps when an LM is frozen downstream; we don't have one. |

### LR sensitivity vs single-query
**Empirical reasoning (no V-JEPA-specific published evidence):** multi-query probes are *less* LR-sensitive than single-query because the 16 queries provide implicit ensembling at the feature level. In practice keep Meta's grid unchanged; decouple LoRA LR by ÷4 to ÷10.

### Reference implementations
- `facebookresearch/vjepa2/src/models/attentive_pooler.py` (single-query, depth=4, heads=16).
- `facebookresearch/jepa/src/models/attentive_pooler.py` (V-JEPA 1; same module, slightly older).
- HF `transformers/models/blip_2/modeling_blip_2.py` `Blip2QFormerModel` (32 learnable queries; `num_query_tokens=32` default, MLP ratio 4, BERT-style cross-attention with shared self-attention).
- Flamingo Perceiver Resampler (third-party reimplementations exist in `lucidrains/perceiver-pytorch` and similar; the original is closed). 64 queries.

---

## Real-SSv2 Import + 4-Frame Transformation (Q2)

### Dataset access status (2025–2026)
- **Canonical host:** `qualcomm.com/developer/software/something-something-v-2-dataset` and `developer.qualcomm.com/downloads/20bn-something-something-download-instructions`. Free for research, registration required. (Qualcomm acquired Twenty Billion Neurons; the qualcomm.com page is the live mirror, formerly 20bn.com.)
- **Mirrors:** HuggingFace `HuggingFaceM4/something_something_v2` (metadata only — 43.7 kB; videos via Qualcomm); academic mirror at `hyper.ai/en/datasets/17204`; OpenDataLab via `mim download mmaction2 --dataset sthv2`.
- **Format:** 19.4 GB total, 20 × ~1 GB tar parts, webm/VP9, files numbered 1..220847. Counts: 220,847 total = 168,913 train + 24,777 val + 27,157 test.
- **License:** academic/research use, no redistribution. Track-B explicitly allows public external data, so this is in-bounds.
- **Annotations:** `train.json`, `validation.json`, `test.json`, `labels.json` (174 id↔name mapping). JSON schema: `[{"id": "<video_id>", "label": "<template with [something] placeholders>", "template": "<template>", "placeholders": [...]}]` (per HF SSv2 dataset card example: `{"video_id": "41775", "text": "moving drawer of night stand", "label": 33, "placeholders": ["drawer", "night stand"]}`).

### Pipeline (step-by-step)
```
1. Download annotations + 20 webm tar parts. Verify md5sums.
   cat 20bn-something-something-v2-?? | tar -xzvf -

2. Load labels.json → class_name → SSv2 class_id (174-way).

3. Filter to your 33 target classes (use existing local class → SSv2 174-way
   mapping; honour the (18,19) Pull L↔R remap and the missing class 27).

4. Read train.json + validation.json. Build candidates:
      [(video_id, ssv2_label) for clip in train+val
       if ssv2_label in target_33_classes]

5. **Test-set contamination filter (CRITICAL).**
   Load validation.json AND test.json video IDs.
   For each Kaggle test sample, compute dHash (64-bit) of its 4 frames.
   For each candidate SSv2 clip, decode the same 4 frames after the 60%/4-frame
   transform, compute dHash. Drop candidate if any frame-dHash Hamming < 5.
   Also exclude all SSv2 val-IDs as a safety net (costs ~14% of candidates).

6. Deduplicate against local train: same dHash test against existing data/train.

7. **4-frame transformation (reverse-engineered hypothesis):**
   For each SSv2 webm:
     a. ffmpeg-decode all frames at native fps; get N_total.
     b. frames_60 = frames[:int(0.6 * N_total)]
     c. idx = np.linspace(0, len(frames_60) - 1, 4).astype(int)
     d. Resize to 256×256 (or your local resolution).
     e. Save as 4-frame stack matching local format.
     f. Optionally emit alt samplings (first-50%, first-70%, full uniform-4)
        for ablation.

8. At train time, apply your existing 4→16 uneven duplication (3,5,5,3) exactly.
```

### Why "first 60% then 4 frames" is plausible
SSv2 clips average ~3.5 s and the action peak typically completes by 50–70% of clip length (rest is held-end framing). Truncating at 60% before subsampling biases toward the active part of the action without losing the climax — consistent with several published head-biased SSv2 sampling recipes and avoids post-action frames that dilute the signal for verbs like "drop into", "pour into". No paper names "first 60% then 4 frames" as canonical, so treat as a working hypothesis; validate empirically.

### Distribution-matching diagnostics
- **Frame-to-frame pixel correlation** (mean abs diff between consecutive of the 4 frames). Plot histograms; synthesised should overlap local. If divergent, sweep cut ∈ {40%, 50%, 60%, 70%, 80%} and pick the value with the lowest 1D Wasserstein distance to local-data's distribution.
- **Per-class motion magnitude** (mean optical-flow magnitude across 3 transitions).
- **A small linear probe** trained on V-JEPA features of synthesised data should achieve roughly the same per-class accuracy as one trained on local data; class divergences > 3 pp flag a sampling mismatch.

### Risks
1. **Contamination is the dominant catastrophic-failure risk.** Mitigation: (a) dHash Hamming < 5 filter; (b) blanket exclusion of all SSv2 val-IDs as safety net.
2. **Class-mapping drift**: the SSv2-174 label string for a target class may not match your local class exactly. Verify by inspecting 5 sample local clips per class against the SSv2 template.
3. **License**: research use allowed; no redistribution. Track-B rules permit external data.

---

## LR-Sensitivity Diagnostic Recipe (Q3) — One Overnight Window

```
Night 1 (~12 h on a 3090):
  fix: 16-query / depth-2 / heads-16 / drop=0.1 probe (Exp 1 architecture)
       data = train minus holdout_clean.json
       epochs = 4 (early-stop on flat val)
  sweep: 12 configs serial
    probe_LR ∈ {3e-3, 1e-3, 3e-4}
    LoRA_LR_ratio (probe_LR / LoRA_LR) ∈ {1, 4, 10, ∞}   # ∞ = LoRA frozen
    WD = 0.05; batch=4, grad_accum=8 → eff. 32, bf16
  log: clean-holdout top-1, train top-1 (gap = capacity vs overfit)
  decision: pick top 3 (probe_LR, ratio) by clean-holdout

Night 2 (~12 h):
  for each of top 3: full 10-epoch training with WD ∈ {0.01, 0.1}
    save ckpts every 200 steps
  → 6 full trainings. Pick best by clean-holdout.

Night 3 (greedy soup, ~30 min):
  for each of top 3, take all 200-step head ckpts (~10 each)
    greedy add to soup if clean-holdout does not decrease
  → final probe.
```

**Expected:** if landscape is flat in `[1e-3, 3e-3]` at ratio 4, you've eliminated a tuning axis; if spiky, you've found the operating point. +0.3 to +1.0 pp from being on the LR optimum.

---

## Pair / Sink Calibration Recipe (Q4) — Top-3 Pairs

### Two-stage: loss-time + post-hoc

**Stage A — loss-time (during Exp 1 training):**
1. **Logit adjustment (Menon 2021).** Replace CE with `CE(z + τ · log p̂(y))`, τ = 1.0; one line of code, Fisher-consistent for balanced error.
2. **Pair margin.** For the 8 enumerated pairs, when model's top-2 includes the wrong pair-mate, add additive margin m on correct-class logit:
   ```
   for y_true in batch:
       partner = PAIR_MAP.get(y_true, None)
       if partner is not None and logits[partner] > logits[y_true] - m:
           logits[y_true] += m
   ```
   Sweep m ∈ {0.1, 0.2, 0.3, 0.5} on clean holdout.
3. **Auxiliary "is_pretend" BCE head**, λ = 0.2, BCE on `{14, 16, 17} → 1, else → 0`.

**Stage B — post-hoc (after training, no re-training):**
1. **Per-class temperature scaling.** Fit one T_c per class on clean holdout by minimising NLL while constraining no class's NLL to *increase*. Sink classes (14, 16, 22, 30) get T_c > 1; strong classes (12, 18, 19) get T_c ≈ 1.
2. **Pair-specific Platt scaling.** For each of the 8 pairs, fit a 2-parameter sigmoid on `z_a − z_b` against the binary "is it a?" label on clean holdout. Apply only when top-2 is exactly that pair.
3. **Sinkhorn / OT calibration (optional, +0.1–0.3 pp at most):** project predicted-probability marginals onto known prior simplex via Sinkhorn iterations; skip unless you have evidence test is non-stratified.

**Anti-overfit guard:** all calibration on `holdout_clean.json` only. If a pair's calibration *decreases* clean-holdout accuracy on the other 31 classes by > 0.1 pp, reject.

---

## Anti-Patterns (Track-B Specific)

### Confirmed dead-ends from your own data — DO NOT REVISIT
- **Any TTA variant** (flip, dense multi-view, multi-crop, 3-crop spatial). You measured −10 pp. SSv2 labels are direction-encoded; flip semantically inverts 18↔19 and others. Even auto-remap doesn't recover the loss.
- **Single-backbone ensembles.** Five V-JEPA 2 ViT-L members with identical backbone is the worst-case ensemble: optimizer collapses onto 2 (your data).
- **Train-time horizontal flip without class-pair remap.** Silently corrupts labels for direction-encoded classes.
- **Tuning anything on the contaminated 80.9% holdout.** It is 10 pp optimistic; any greedy-soup or LR pick on it silently picks LB-overfit checkpoints.
- **Pretraining a backbone from scratch on SSv2.** Out-of-scope; wouldn't fit in 10 days anyway.
- **VideoMAEv2 pretraining recipes.** Out-of-scope per user.

### Public-LB-overfit anti-patterns — held-out test is the real eval
- **Pseudo-labeling Kaggle test** and training on pseudo-labels. Classic LB-overfit; held-out test split likely has different distribution and noise compounds.
- **Per-class temperature scaling fit on public LB feedback.** Overfits LB probes directly into model. Calibration MUST be fit on `holdout_clean.json`.
- **Ensemble weight fitting on public LB.** Same problem.
- **Reading per-class accuracy from public LB.** Don't.
- **Multiple submissions to "probe" specific predictions.** Each is a degree of freedom you lose on held-out.
- **Choosing the LB-best checkpoint rather than clean-holdout-best.** Your 70.52 LB is a one-sample estimate with std ≈ √(p(1−p)/N) ≈ 1.7 pp at N~1500; choosing on LB is noise-chasing.

---

## Recommendations (staged, with thresholds)

### Days 1–2 (must-do prerequisite)
- Build `holdout_clean.json` (Exp 8). **Threshold:** if current 70.5% LB ≠ clean-holdout top-1 within ±1.5 pp, your data pipeline has an issue — fix before continuing.
- Download SSv2 webm + annotations from Qualcomm. Build 4-frame transform pipeline (Exp 3 steps 1–4).

### Days 3–4 (highest-EV)
- Run Exp 1 coarse pass: multi-query (16) cross-attention probe + 5 LRs × 1 WD = 5 runs overnight. **Threshold:** if best LR gives < +0.5 pp on clean holdout vs single-query baseline, multi-query is not your problem — go to Exp 2 / Exp 3 instead.

### Days 5–6
- Run Exp 2 (logit adjustment + pair margin) on top of Exp 1's best. **Threshold:** if confusion matrix doesn't show meaningful drop on (11,14), (16,22), (9,11), revert.
- In parallel, finish Exp 3 (real-SSv2 import) and train one model on combined corpus. **Threshold:** if clean-holdout per-class recall on classes 16, 11, 17 doesn't go up by ≥ 5 pp each, your 4-frame transform doesn't match — try alternative samplings.

### Days 7–8
- Run Exp 4 (probe-only continuation + greedy soup).
- Run Exp 6 (LR-sensitivity diagnostic) if Exp 1 didn't already settle it.

### Days 9–10
- Run Exp 5 (architecturally diverse ensemble). Download VideoMAE-base-ssv2 (70.6% top-1 per HF card), TimeSformer-hr-ssv2; mask 174-way heads to 33 classes; run inference; fit weights on `holdout_clean.json` with non-negative simplex constraint. **Threshold:** if no new member > 0.55 clean-holdout, drop it.
- Final LB submission from greedy-soup of top-3 architecturally-distinct models, weights fit only on `holdout_clean.json`.

### Decision triggers to change plan
- If after Day 4 the multi-query probe gives < +1 pp, **abandon architecture changes**; spend Days 5–10 on Exps 2 + 3 + 5.
- If real-SSv2 contamination check flags > 5% overlap with Kaggle test, **abandon Exp 3** — the test set is too leaky to trust.
- If clean holdout != LB ± 1.5 pp after Day 2, **stop and debug**; further experiments are misdirected.

---

## Caveats

- The "first 60% then 4 frames" recipe is a **hypothesis**, not a documented standard. The professor may have used a different transformation (uniform 4 frames, first 50%, varying head-bias). The distribution-matching diagnostics are the empirical safeguard.
- The single-learnable-query claim for V-JEPA 2's attentive pooler is verified from the published config (`num_probe_blocks: 4`, `num_heads: 16`, no `num_queries` parameter exposed in `configs/eval/vitl/ssv2.yaml`) and from the `AttentiveClassifier(embed_dim, num_heads=16, depth=4, num_classes=174)` instantiation in the V-JEPA 2 demo notebook (no `num_queries` kwarg passed → module default). The module default is `1` per the equivalent `AttentivePooler` in `facebookresearch/jepa/src/models/attentive_pooler.py`. If on inspection `num_queries > 1` is the actual default, your multi-query experiment is a *parameter sweep* rather than a *new architecture* — same wall-clock, same prior, same expected gain.
- All pp-gain numbers above are **expected ranges from literature + your confusion matrix**, not point predictions. Threshold-based decision rules are the safety net.
- The contamination-overlap risk for the SSv2 import is **real and the dominant single source of catastrophic failure** for Exp 3. The dHash filter (Hamming < 5 / 64) is conservative but not bullet-proof; the secondary "exclude all SSv2 val IDs" rule is the safety net at the cost of ~14% of importable clips.
- Greedy-soup gains in the literature (Wortsman et al. 2022 ViT-G ImageNet: 90.78 → 90.94, +0.16 pp) are reported mostly for *full fine-tuning*, not PEFT/LoRA. Probe-only soup (Exp 4) is the safest bet; LoRA-weight souping is less well-studied — try probe-only first.
- Track-B rules allow external pretrained backbones and public external data — confirmed in task framing — so Exp 3 (real SSv2) and Exp 5 (HF SSv2-finetuned checkpoints from MCG-NJU and Meta) are in-bounds.
- The Wortsman et al. greedy-soup result is most rigorously demonstrated on full fine-tuning of CLIP ViT-B/32 with a held-out validation set (Figure 1); applying it to LoRA + small held-out is in-spirit but not directly evidenced in the original paper.
- InternVideo2 community evaluations (HF discussion on `OpenGVLab/InternVideo2-Stage2_6B`) note an ~10-point gap to paper-reported retrieval numbers because reported results include an ITM re-ranking stage; for classification on SSv2 the published `1B_ft_ssv2_f8.sh` script with `--lr 1e-4 --drop_path 0.3 --layer_decay 0.915` is the recipe to follow, but expect a head-only / probe-only adaptation on a 3090 because the 1B model is too large for end-to-end fine-tuning on 24 GB.