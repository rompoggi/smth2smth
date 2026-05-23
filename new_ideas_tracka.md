# Deep Literature Review — Closing a 4.5-pt Gap on the SSv2-33 Closed-World Challenge

## TL;DR
- **Spend the week on (a) learned-weight logit ensembling of 3–5 diverse VideoMAE-v2 fine-tunes, and (b) a Perceiver/attentive-probe head over a *partial* TimeSformer-style temporal modification of the last 6 ViT-B blocks** — both are documented in the literature (Bulat NeurIPS 2021; Bardes V-JEPA TMLR 2024; Caruana ICML 2004) and together explain the competitors' 4–5 pt advantage with high confidence.
- **Do not chase more SSL epochs.** Per the official MCG-NJU Hugging Face model cards, going from 800 → 2400 SSL epochs on SSv2 moves ViT-B Top-1 only from 69.6 → 70.6 (+1.0 pt total over a 3× compute increase); the original VideoMAE Figure 5 shows the curve has already flattened by 800. Your 200 → 250 ablation is unlikely to move > ~0.5 pt; stop SSL at the next plateau and reallocate compute to fine-tuning seeds.
- **Fix two near-certain overfitting bugs first**: your `drop_path=0.1` (vs official 0.2), `frame_mixup α=5` (non-canonical and Beta(5,5)-style aggressive mixing), and `num_sample=1` (vs official repeated-aug `=2`) explain part of the 31-pt train/val gap; restoring the official MCG-NJU recipe is the cheapest 1–2 pt available.

## Key Findings

1. **The "spatial-only early, spacetime late" pattern Stan used is documented exactly once in published video-transformer literature — Bulat et al. NeurIPS 2021 (XViT)** — and the empirical finding there is that *the position of temporal layers does not matter, only the count*: on SSv2 ViT-B/16, applying space-time mixing to only the first half scored 61.7 Top-1, second half 61.6, all layers 62.6 (Δ ≈ 1 pt). The Bulat paper's verbatim conclusion: *"the exact layers within the network that self-attention is applied to do not matter; what matters is the number of layers it is applied to."* This means Stan's 51.08% is **not** primarily explained by the partial-depth split per se — it is explained by *having any temporal attention at all on top of a CLS/avg-pool baseline*.

2. **The Perceiver-style cross-attention pooling head has strong, recent published support for video classification.** V-JEPA (Bardes et al., TMLR 2024, arXiv 2404.08471) Table 3 reports **+17.3 Top-1 points on Kinetics-400** when replacing average pooling with a cross-attention "attentive probe" head — by far the largest single-component delta in their ablation. V-JEPA 2 (Assran et al., 2025, arXiv 2506.09985) keeps this head and reaches **75.3% Top-1 on SSv2 with ViT-g at 256px (77.3% at 384px)** vs 69.7% for InternVideo and 55.4% for PEcoreG under the same frozen-encoder + attentive-probe protocol (Table 4). This is the lever most likely responsible for the bulk of Stan's gain over a CLS/attentive-probe baseline.

3. **Learned-weight logit ensembling is well-grounded in Caruana ICML 2004 (ensemble selection) and Sill et al. 2009 (FWLS, Netflix-Prize 2nd place), but the +3 pt single-component gain Baptiste claims is at the *upper* edge of published deltas.** Model Soups (Wortsman et al., ICML 2022, arXiv 2203.05482) Table 1 shows greedy soup vs best single ViT-G/14 model is only +0.16 pt on ImageNet (90.94% vs 90.78%); KDD-cup 2014 stacking gave ~+2 AUC pt (van Veen Kaggle ensembling guide); Caruana 2004 normalized gains over the best single model are typically ~0.1–0.2. A +3 pt gain is plausible only if base models are sufficiently *diverse* (different heads, augmentations, frame samplings — not just different seeds).

4. **Your SSL pretraining is past the steep part of the curve.** VideoMAE Tong et al. (NeurIPS 2022) Figure 5 reports SSv2 fine-tune Top-1 of 66.4 → 67.9 → 69.6 → 70.3 → 70.6 at 200/400/800/1600/2400 epochs. The marginal gain from 200 → 250 epochs on a 4×-smaller dataset is bounded above by ~0.3 pt; even 200 → 800 would be ~3 pt at most. The prof's epoch ablation is methodologically correct but unlikely to be the highest-EV move this week.

5. **VideoMAE's *official* SSv2 fine-tuning recipe — which your config diverges from in three ways — is:** AdamW, lr 5e-4, weight_decay 0.05, **drop_path 0.2** (not 0.1), epochs 40–50, **num_sample=2** (repeated augmentation), Mixup α=0.8, CutMix α=1.0, RandAug `rand-m7-n4-mstd0.5-inc1`, layer-wise lr decay 0.75, label smoothing 0.1, MultiScaleCrop, 16 frames @ 224, test 2×3 views. Your `frame_mixup α=5` is *not* a published technique I can identify — likely an in-house variant that may be hurting via too-aggressive (near-50/50) frame mixing.

## Details

### Research Question 1 — Factorized Space-Time Attention + Perceiver Pooling Head

**(a) Partial-depth temporal-attention insertion.**
The full TimeSformer (Bertasius, Wang, Torresani, ICML 2021, arXiv 2102.05095) applies *divided* space-time attention in **every** block — temporal then spatial within each of the 12 ViT-B/16 blocks. Reported SSv2 ViT-B Top-1, 8 frames @ 224: **59.1 / 85.6** (official model zoo, github.com/facebookresearch/TimeSformer); TimeSformer-HR (16×448) 61.8; TimeSformer-L (64×224) 62.0. ViViT (Arnab et al., ICCV 2021, arXiv 2103.15691) similarly factorises *every* block (Model 3 "factorised self-attention" — spatial then temporal sub-block inside each transformer layer), or factorises encoder-wise (Model 2: spatial encoder then temporal encoder stacked end-to-end). ViViT-B Factorised Encoder reaches **65.4% SSv2 Top-1** with full regularisation (paper Table 3, verbatim: *"from 60.4% to 65.4%, on SSv2 by using all the regularisation in Tab. 3"*). MViT / MViTv2 (Fan CVPR 2021 / Li CVPR 2022) use pooling attention throughout but at hierarchically varying stride; MViT-B 64×3 reaches 67.7 SSv2, MViTv2-L 73.3. Video Swin (Liu et al., CVPR 2022) uses 3D window attention throughout. None of these is partial-depth.

The single paper explicitly ablating partial-depth temporal layers is **Bulat, Pérez-Rúa, Korbar, Martínez, Tzimiropoulos "Space-time Mixing Attention for Video Transformer," NeurIPS 2021, arXiv 2106.05968 (XViT)**. From their Table 2a on SSv2 with ViT-B/16, 8 frames:
- baseline (no temporal mixing, tw=0) — 45.2 Top-1
- temporal in 1st-half blocks only — **61.7 / 86.5**
- temporal in 2nd-half blocks only — **61.6 / 86.3**
- temporal in alternating ("odd-position") blocks — 61.2 / 86.4
- temporal in all blocks — **62.6 / 87.8**

Full XViT (all-layer space-time mixing + 1 extra temporal-attention layer) reaches **64.4% SSv2 Top-1, 8 frames, 1×3 views; 64.5 with 16 frames**.

**Interpretation.** Stan's "last-6-of-12 only" choice is well-motivated and saves compute (~half the temporal-attention FLOPs), with a documented penalty of ≤ ~1 pt versus all-12 — exactly consistent with NeurIPS 2021 evidence. Importantly, the absolute jump from "spatial-only ViT" → "any partial-depth space-time" is ~16 pt in Bulat's setup — this is most likely where Stan's biggest delta comes from.

**Also relevant — adapter-style temporal injection in only some blocks:**
- **AIM (Yang et al., ICLR 2023, arXiv 2302.03024)**: freezes a pretrained ViT and inserts spatial, temporal, joint adapters; the temporal adapter reuses pretrained spatial-attention weights for temporal modeling. Reports **87.5% Top-1 on Kinetics-400** with ViT-L/14 (CLIP-pretrained, 16 frames) and 11M tunable params for ViT-B — the AIM paper's headline accuracy is for K400, not SSv2.
- **ST-Adapter (Pan et al., NeurIPS 2022)** and **EVL (Lin et al., ECCV 2022, arXiv 2208.03550)** insert temporal modules in every block of a frozen CLIP backbone with a Transformer decoder + learned query token over frame-level features. EVL's "learn a query token to dynamically collect frame-level spatial features" is conceptually a Perceiver-style head.
- **DiST (Qing et al., ICCV 2023, arXiv 2309.07911)** uses a dual encoder: frozen ViT spatially + lightweight temporal encoder.

**(b) Perceiver/learned-query pooling head.**
The Perceiver (Jaegle et al., ICML 2021, arXiv 2103.03206) and Perceiver IO (Jaegle et al., ICLR 2022, arXiv 2107.14795) introduced the pattern: K learned latent queries cross-attend to a long key/value sequence, decoupling depth from input size. The original Perceiver reports video on AudioSet only (43.2 mAP video-only), not SSv2.

**The directly relevant evidence for video classification is V-JEPA / V-JEPA 2:**
- Bardes, Garrido, Ponce, Chen, Rabbat, LeCun, Bojanowski, Assran, Ballas, "Revisiting Feature Prediction for Learning Visual Representations from Video," TMLR 2024, arXiv 2404.08471. Table 3: replacing average pooling on frozen V-JEPA features with an attentive (cross-attention) probe head **gains +17.3 Top-1 on Kinetics-400** (verbatim: *"Using adaptive pooling with a cross-attention layer leads to improvements of +17.3 points on K400"*).
- V-JEPA 2 (Assran et al., 2025, arXiv 2506.09985) uses the same head: "the last block replaces self-attention with cross-attention using a learnable query token." SSv2 75.3 Top-1 (ViT-g/256) and 77.3 Top-1 (ViT-g/384). Table 4 verbatim: *"It achieves a top-1 accuracy of 75.3 on SSv2 compared to 69.7 for InternVideo and 55.4 for PEcoreG."*

**This is the single largest documented architectural lever for closing a CLS-pool vs attention-pool gap on video.** Mean-pooling the K=16 Perceiver queries before the linear classifier (as Stan does) is a minor variant; the published gain is from the cross-attention itself.

**(c) Why 4 frames is competitive.** SSv2 generally benefits from more frames (VideoMAE-B uses 16×2, official config). However: (i) on a 33-class subset, classes with strong temporal-direction cues may collapse, weakening the marginal value of more frames; (ii) with an explicit divided-space-time attention in the *late* blocks plus a Perceiver head, the 4 frames are processed with temporal mixing at every spacetime block, whereas a vanilla ViT-B + attentive probe over 4 frames has no temporal mixing inside the encoder at all. Frame Flexible Network (FFN, Tan et al., CVPR 2023, arXiv 2303.14817) reports a frame-mismatched evaluation gap of 7.08/5.15/2.17 pt at 4/8/16 frames vs separately trained models on SSv1 — i.e., the architectural advantage at 4 frames can exceed the raw frame-count gain. That said, a 51.08 result with 4 frames and ~50k videos is consistent with both "architecture explains it" and "noise / lucky test-set fold"; you cannot fully separate these without a multi-seed run.

**(d) Implementation pointers.**
- **TimeSformer divided-attention reference**: facebookresearch/TimeSformer (`timesformer/models/vit.py`, `attention_type='divided_space_time'`); the divided-attention block is ~50 lines and easy to slice into "spatial-only" vs "spacetime" variants.
- **VideoMAE backbone**: MCG-NJU/VideoMAE `modeling_finetune.py` — copy `Block` → add a `SpaceTimeBlock` that reuses spatial attn and adds a `temporal_attn` `nn.MultiheadAttention` over the time axis with residual, then replace blocks 6–11 in `VisionTransformer`.
- **Perceiver / cross-attention head**: lucidrains/perceiver-pytorch, or copy V-JEPA's attentive probe (`jepa/src/models/attentive_pooler.py` in facebookresearch/jepa). Hugging Face `transformers.PerceiverModel` includes `PerceiverClassificationDecoder` but you want the **AttentivePooler** pattern: K learned queries × 1 cross-attn layer × mean-pool → linear.
- 2-3 day budget: realistic. The blocks change is ~100 lines; the head is ~50 lines; weight-load surgery requires care because the new temporal-attn layers in blocks 6–11 are randomly initialised (warmup at low lr, or zero-init the temporal-attn output projection à la AIM/AdaLN).

### Research Question 2 — Learned-Weight Logit Ensembling

**(a) Formal name.** This is **stacked generalization** (Wolpert, *Neural Networks* 1992) with a constrained linear combiner; specifically, a linear stacking model with one scalar weight per base model. The two-stage "freeze θ, then refit base models on train+val" protocol is unusual — the canonical stacking protocol uses out-of-fold predictions (so base models never see the validation set). Baptiste's variant is closer to **greedy ensemble selection** (Caruana, Niculescu-Mizil, Crew, Ksikes, ICML 2004, "Ensemble Selection from Libraries of Models") where weights are learned on a held-out validation set and the base models stay frozen, except that Caruana selects integer counts via forward-stepwise selection rather than learning continuous θ via cross-entropy.

The "refit on train+val with frozen θ" is a pragmatic competition hack — equivalent to assuming θ is invariant to the data-volume increase, which is reasonable when base models are well-calibrated and the validation set is representative.

**(b) Scalar vs vector weights, constraints.** Sen & Erdogan (arXiv 1106.1684, "Max-Margin Stacking and Sparse Regularization for Linear Classifier Combination and Selection") compare:
- **Weighted Sum (WS)** — one scalar per base model (Baptiste's setup),
- **Class-dependent Weighted Sum (CWS)** — θ ∈ R^(N×C),
- **Linear Stacked Generalization (LSG)** — full linear map θ ∈ R^(N·C × C).

Findings: CWS and LSG add parameters proportional to C, and need more held-out data to avoid meta-overfitting. With 33 classes and a small validation set, **scalar weights are the safer choice**. Common constraints: non-negativity (θ_i ≥ 0) and softmax simplex (Σθ_i = 1) regularise the meta-learner; in scipy.optimize.minimize with SLSQP, this is one line.

**(c) Diversity sources.** Surveyed evidence:
- **Random-seed-only diversity** (deep ensembles, Lakshminarayanan 2017) gives ~+1-2 pt on ImageNet for 5 models — your floor.
- **Architecture diversity** (ResNet + ViT, etc.) — typically +2-4 pt over seed-only on ImageNet (Hyperdeep ensembles, Wenzel ICML 2020).
- **Head diversity (same backbone, different heads)** — documented gains are smaller (+0.5-1 pt; predictions are highly correlated when backbone is shared). Multi-head ensembles (Lee et al., "Why M Heads are Better than One," arXiv 1511.06314) show modest gains under shared backbone.
- **Augmentation/sampling diversity** — TTA-style: different temporal samplings (clip starts) and crops can add 0.5-1 pt at inference time alone.
- **Snapshot ensembles** (Huang et al., ICLR 2017, arXiv 1704.00109) give cheap diversity from cyclical lr but typically less than deep ensembles.

**Strong recommendation: prioritize architecture/recipe diversity over head diversity.** For 3 models in 1 week: (model A) VideoMAE-v2 ViT-B + attentive probe, 16 frames; (model B) VideoMAE-v2 ViT-B + Perceiver head + late-block spacetime attention, 8 frames; (model C) VideoMAE-v2 ViT-B + different augmentation seed, 16 frames with repeated-aug. Refit θ on val.

**(d) Marginal returns.** Empirical curve from Caruana 2004 and Model Soups (Wortsman ICML 2022, arXiv 2203.05482): 2 models → ~70% of full gain, 4 models → ~90%, 8+ models → diminishing returns. For a 1-week budget, **3-5 models is the sweet spot**.

**(e) Comparison to simpler baselines.** Reported deltas:
- Equal logit-averaging vs single-best: typically +1-2 pt (the universal baseline).
- Learned-weight vs equal averaging: Model Soups Table 1 shows greedy soup vs best single ViT-G/14 is **+0.16 pt on ImageNet (90.94% vs 90.78%)**; Caruana 2004 normalized gains are 0.1-0.2; KDD-cup 2014 stacking ~+2 AUC pt.
- Softmax-then-average vs logit-average: marginal, ~0.1-0.3 pt — logit-average is the safer default.
- **The +3 pt single-component gain Baptiste claims is at the upper edge of published deltas.** It is plausible only if (i) the base models are well-diverse, or (ii) one base model is much weaker than the others and learned weights effectively down-weight it, or (iii) noise on the public LB. **I would budget for +1-2 pt from this lever in expectation**, with +3 as an upside.

**(f) Open-source implementations.**
- `scipy.optimize.minimize(cross_entropy, x0=ones/N, method='SLSQP', constraints=[...])` — 20 lines, what most Kagglers use.
- `sklearn.ensemble.StackingClassifier` — works but assumes meta-learner is a sklearn estimator.
- `mlens` (Sebastian Flennerhag) — supports K-fold blending properly.
- `pyensemble` (dclambert) — implements Caruana ICML 2004 greedy selection.
- Kaggle reference notebooks: `tolgadincer/ensemble-weight-optimization`, `daisukelab/optimizing-ensemble-weights-using-simple`.

### Research Question 3 — VideoMAE Pretraining-Epoch Ablation

**(a)–(b) Published epoch-vs-accuracy curve.** From the original VideoMAE NeurIPS 2022 paper (Tong, Song, Wang, Wang, arXiv 2203.12602), Figure 5(a), SSv2 Top-1 with ViT-B as a function of SSL pretraining epochs:

| Epochs | 200 | 400 | 800 | 1600 | 2400 |
|--------|------|------|------|------|------|
| SSv2 Top-1 (%) | 66.4 | 67.9 | 69.6 | 70.3 | 70.6 |

Independently confirmed by the Hugging Face model cards: `MCG-NJU/videomae-base-short-finetuned-ssv2` (800 epochs) reports 69.6% Top-1; `MCG-NJU/videomae-base-finetuned-ssv2` (2400 epochs) reports 70.6% Top-1 (verbatim: *"This model obtains a top-1 accuracy of 70.6 and a top-5 accuracy of 92.6 on the test set of Something-Something-v2"*).

**The curve is sub-logarithmic and flattens hard after 800 epochs** — each doubling buys ≤0.5 pt past that point. VideoMAE v2 (Wang et al., CVPR 2023, arXiv 2303.16727) used dual masking to reduce per-epoch cost but the *epoch-vs-accuracy* curve shape is essentially identical.

**For your setup (200 epochs, 4× smaller dataset, 33 classes):** the saturation should occur *earlier* in epochs (smaller dataset → faster convergence) but the asymptote should be *lower* (less data variety). I would forecast your 200→250 ablation gives +0.2-0.5 pt at most, and 200→400 gives +0.5-1.0 pt.

**(c) SSL loss ↔ downstream accuracy.** No paper has cleanly established that MAE reconstruction loss is a faithful proxy for downstream accuracy; in fact, several MAE/MIM papers note the relationship is weakly monotone but noisy. The loss curve flattens *before* downstream gains saturate. You cannot predict downstream gain from loss alone — you must FT to know.

**(d) Pretraining on train+val+test vs train+test.** VideoMAE (NeurIPS 2022) Section 4.4 finding (iii): *"data quality is more important than data quantity for SSVP. Domain shift between pre-training and target datasets are important issues."* Including val in SSL pretraining is essentially free (val labels are not used) and removes a small domain-shift; **expected gain ~0.1-0.3 pt**, not large but cheap to do.

**(e) Alternative MAE objectives.**
- **MotionMAE** (Yang et al., 2022, arXiv 2210.04154): adds motion-structure prediction; +1.2 pt over VideoMAE on SSv2 with ViT-B (75.5% in domain-specific pretraining). Requires modifying SSL loss; **non-trivial to add in 2 days**.
- **MGM / Motion-Guided Masking** (Fan et al., 2022): motion-guided masking instead of random tube; +1-2 pt on SSv2 with ViT-B. Optical-flow needed.
- **MGMAE** (Huang et al., ICCV 2023, arXiv 2308.10794): online optical-flow + mask warping; +0.7-1 pt on SSv2.
- **SiamMAE** (Gupta et al., NeurIPS 2023): siamese MAE for correspondence; reported on VOS, not SSv2 classification.
- **MVD** (Wang et al., CVPR 2023, arXiv 2212.04500): masked feature prediction with both image-MIM and video-MIM teachers; +2.4 pt over VideoMAE on SSv2 with ViT-L (76.7%). Needs teacher models.
- **UMT (Unmasked Teacher)** and **VideoMAE-Distilled** (CVPR 2023 supplemental): need image-text teacher; not feasible in your closed-world setup.
- **MAM² / MAM-squared** (arXiv 2210.05234): masked appearance-motion modeling; +0.7 pt on SSv2, **2× faster pretraining** (400 epochs ≈ VideoMAE 800). **This is the most attractive drop-in if you want to try a new SSL objective in <2 days.**
- **MOFO** (arXiv 2308.12447): motion-focused SSL; reports +4.7 pt on SSv2 over VideoMAE. Optical-flow needed.

**Bottom line on RQ3: more SSL pretraining is low-EV.** Document the 200/250 ablation for the report (the prof asked), but allocate the bulk of remaining compute to fine-tuning and ensembling.

### Research Question 4 — Data-Augmentation Recipe for VideoMAE FT on SSv2

**(a) Canonical official recipe (MCG-NJU/VideoMAE/FINETUNE.md and run_class_finetuning.py defaults for SSv2):**
- model: `vit_base_patch16_224`
- optimizer: AdamW, β=(0.9, 0.999)
- lr: 5e-4 (with linear warmup 5 epochs, cosine decay)
- weight_decay: 0.05
- layer_decay: 0.75
- epochs: 40 (paper) to 50 (later config)
- batch_size: 64–128 effective (per-GPU 8 × 64 GPUs in the official run)
- **num_sample: 2 (repeated augmentation)**
- **drop_path: 0.2** (ViT-B)
- num_frames: 16, tubelet_size: 2
- input_size: 224, short_side_size: 224, MultiScaleCrop scale [0.08, 1.0], aspect [0.75, 1.333]
- random_horizontal_flip: **False on SSv2** (left-right matters for actions — explicit in `ssv2.py`: `random_horizontal_flip=False if args.data_set == 'SSV2' else True`)
- RandAug `rand-m7-n4-mstd0.5-inc1`
- mixup α: 0.8, cutmix α: 1.0, mixup_prob: 1.0, mixup_switch_prob: 0.5
- label_smoothing: 0.1
- random_erase prob: 0.25, mode 'pixel', count 1
- test_num_segment: 2, test_num_crop: 3 (2×3 views)

**Your config deviates on**: `drop_path=0.1` (should be 0.2), `num_sample=1` (should be 2), `RandAug mag 9 / 2 ops` (should be `m7/n4`), and `frame_mixup α=5` (non-canonical).

**(b) Repeated augmentation (num_sample=2).** The VideoMAE MODEL_ZOO explicitly flags repeated-augmentation as a positive — the 800-epoch ViT-B SSv2 checkpoint is annotated "(w/o repeated aug) 69.6" while the 2400-epoch ViT-B with repeated-aug reaches 70.8. While this conflates two variables (epochs + aug), the README's explicit "w/o repeated aug" caveat indicates the authors view repeated-aug as standard for SSv2. The original DeiT paper (Touvron ICML 2021, arXiv 2012.12877) reports DeiT-B 81.8% at 224 with repeated-aug as part of the full recipe but does not isolate a clean repeated-aug delta in Table 7. **Estimated gain on your setup: 0.5-1.5 pt.** Cheap to add.

**(c) Mixup/CutMix α.** Official: 0.8 / 1.0. Frame-level mixup with α=5 is *not* a recognized published technique under that name. Beta(5,5) concentrates mass near 0.5 (near-equal mixing) — much more aggressive than Beta(0.8,0.8) which is heavy-tailed near 0 or 1. **Strong recommendation: revert to α=0.8 (or disable entirely and isolate).**

The closest *published* video-specific mixup is **Selective Volume Mixup (SV-Mix, Tan et al., arXiv 2309.09534)** — learned spatial+temporal selective mixing, reported +1.7 pt over standard Mixup/CutMix on SSv2 with TSM and ViT backbones (the TSM table in the paper shows SSv2: +Cutmix +0.2, +Mixup −0.9, +SV-Mix +1.7). Implementation is non-trivial (~200 lines); only attempt if time allows.

**(d) Temporal augmentations.** Documented gains for VideoMAE-class methods:
- Temporal jitter (random frame index sampling): standard, included in `ssv2.py`.
- Frame dropout: typically marginal (<0.3 pt) on SSv2.
- VIPriors temporal CutMix / FadeMixUp (Kim et al., arXiv 2008.05721): +0.5-1 pt on UCF101 — gains on SSv2 unreported.
- Time-reversal: usually **hurts** on SSv2 because direction matters (e.g., "pushing X" vs "pulling X").
- SV-Mix (above): +1.7 pt.

**(e) "What happens next" anticipation framing.** If the test set is biased toward video beginnings, randomly truncating training videos to a shorter prefix during training is a reasonable, ad-hoc technique with **no canonical published gain on SSv2-style data**. EPIC-KITCHENS anticipation papers do this, but those are anticipation tasks. **Try only if your error analysis shows test clips are systematically shorter or earlier in the video.**

## Recommendations

**Priority-ordered 1-week plan (with stop-conditions and expected gains over your 46.6% baseline):**

| Day | Action | Expected gain | Stop / abort if |
|-----|--------|---------------|------------------|
| 1 | **Fix the FT recipe to official VideoMAE**: drop_path 0.2, num_sample=2, RandAug m7/n4, Mixup 0.8 / CutMix 1.0, drop frame_mixup α=5, no horizontal flip, 16 frames. Run one training. | **+1.5 to 2.5 pt** | If val Top-1 drops, revert one change at a time. |
| 1-2 | **Add Perceiver / cross-attention pooling head** (K=16 learned queries, 1 cross-attn layer, mean-pool, linear) on top of existing 16-frame VideoMAE ViT-B; fine-tune from your SSL checkpoint. | **+1 to 2 pt** | If val gap widens >5 pt vs attentive probe, lower lr on head. |
| 2-3 | **Add late-block divided space-time attention** (blocks 6-11, Bulat-style or TimeSformer-style temporal-attn after spatial-attn) with new temporal-attn output projections zero-initialised. Train. | **+0.5 to 1.5 pt** | Skip if Day 1-2 already at 50%+. |
| 3-5 | **Train 3-5 diverse base models**: (A) 16-frame attentive-probe baseline (Day 1 model); (B) 8-frame Perceiver+late-spacetime (Day 2-3 model); (C) different seed of A with stronger RandAug; (D) optionally 16-frame Perceiver. Hold out 10% as val. | (per-model val: 47-50%) | If any model val Top-1 < 45%, exclude from ensemble. |
| 5-6 | **Learned-weight logit ensembling**: scipy SLSQP on val to find scalar θ_i (non-negative, sum-to-1) minimizing cross-entropy on Σ θ_i L_i(x). Submit equal-average baseline first to measure pure-averaging gain. | **+1 to 3 pt over best single** | If learned weights collapse to equal weights, stop — accept equal averaging. |
| 6-7 | **Retrain base models on train+val with frozen θ**; submit. Also document the 50/100/150/200/250 SSL-epoch ablation (light FT only, no ensembling) so you have it for the report. | **+0.3 to 0.7 pt** (more data) | If LB drops, revert to train-only models with same θ. |

**Total expected gain: 4-7 pt over 46.6% baseline. Target: 50.5-53%.**

**Decision thresholds:**
- **If after Day 1 (recipe fix) you are < 47.5%**: there is a deeper bug (data leakage, label mapping, or temporal-sampling). Halt and debug before adding architecture.
- **If after Day 3 (Perceiver + late spacetime) you are < 49.0%**: skip the new SSL pretraining; double down on ensembling 3-4 architectural variants.
- **If after Day 5 single best > 50.5%**: ensembling will still move the needle by 1-2 pt and you should do it. If single best > 51.5%: ensembling may give only +0.5 pt; consider train+val refit instead.
- **Do not run the 200→250 SSL ablation as a serious gain-seeking experiment** — only run it briefly for the prof's report (one extra checkpoint at 250 epochs FT'd briefly).

## Caveats

1. **The competitors' 50.27% and 51.08% are single submissions on a public Kaggle LB** — with ~50k videos and one test split, ±0.5-1 pt of LB noise is normal. Without seeing private-LB or multi-seed variance, treat the 4.5 pt gap as having ±1.5 pt uncertainty. A well-executed recipe + ensemble may match them even without their exact architecture.

2. **The "partial-depth temporal attention" Stan used is *not* substantially better than full-depth divided attention** per Bulat NeurIPS 2021 — at most ±1 pt difference. Stan's win is more likely from (a) Perceiver head, (b) any temporal attention at all if his baseline lacked it, (c) random variance. Do not over-invest in replicating the exact "first half spatial only" architecture; a uniformly-spaced temporal-attention insertion or even all-block insertion will perform within noise.

3. **The +3 pt Baptiste claims for learned-weight ensembling alone is at the upper edge of published deltas.** Model Soups Table 1 shows only +0.16 pt for greedy vs best-single on ImageNet ViT-G/14; Caruana 2004 gains over best single are 0.1-0.2 normalized; KDD-cup 2014 stacking gave ~+2 AUC pt. Plan for +1-2 pt as the expected value; +3 pt is achievable only with genuinely diverse models. Equal-weight averaging of 3 diverse models often closes 80% of the gap to learned weights.

4. **Including the test-set frames in SSL pretraining is borderline transductive learning.** The closed-world challenge rules permit it (you stated "200-epoch SSL on provided train+test frames"), but be aware the published VideoMAE numbers do *not* use test frames in SSL; comparisons to literature should note this. Including val frames in SSL is unambiguously fine.

5. **A 31-pt train/val gap is extreme.** This implies near-perfect train fit at the current FT config. Even after fixing the recipe, expect train Top-1 = 95-99% and val = 47-51% on this 33-class subset. The cleanest diagnostic: log per-class val accuracy and check if specific classes (e.g., motion-symmetric ones like "moving X up/down") are systematically failing — these may benefit from stronger temporal modeling rather than more augmentation.

6. **The 4-frame vs 16-frame question is unresolved by literature for your specific 33-class subset.** Frame Flexible Network suggests architecture can compensate substantially at 4 frames, but if your 33 classes include direction-sensitive ones, 4 frames may be too sparse. I recommend running both as ensemble members rather than picking one.

7. **No paper directly establishes the +17.3 pt cross-attention vs avg-pool gain transfers to SSv2 with VideoMAE features.** V-JEPA's gain is on K400 with V-JEPA features; the head is also trained with the SSL representation frozen, which is different from VideoMAE's full fine-tuning regime. SSv2 is more temporally demanding; the realistic gain on a fully-fine-tuned ViT-B with VideoMAE pretraining is likely smaller (1-3 pt) rather than 17 pt — the V-JEPA number reflects how much information avg-pool *discards* from frozen features, not how much a Perceiver head adds to a fully-fine-tuned model.