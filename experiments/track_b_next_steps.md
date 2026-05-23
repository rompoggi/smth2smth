# Closing a 6 pp gap on SSv2-33 with V-JEPA 2 ViT-L/256: ranked experiments for a single 3090 in ~10 days

> ⚠️ **READ THIS SECTION FIRST.** The body of this document is the raw Deep Research output (~2026-05-18). After review, several of its claims were corrected against project-specific facts the agent did not have access to. The corrections below override the body where they conflict.

---

## Annotations and corrections (2026-05-18 review)

### A. The (18,19) class-pair remap was CORRECT — keep it

**What the doc says:** §4 ("SSv2 directional pairs"), §150 and §164 (Anti-patterns), and Experiment 3 all claim our TTA remap of `(18, 19)` is wrong because in SSv2's native 174-class index space those IDs map to "Holding ... in front of" and "Holding ... next to" — not mirror images.

**What's actually true:** Our subset folder names use a *local* index prepended to the SSv2 class name (e.g. `020_Putting_something_behind_something`). In our local-index space:

- `018_Pulling_something_from_left_to_right`
- `019_Pulling_something_from_right_to_left`

These ARE the mirror pair — equivalent to SSv2 native `(86, 87)`. The existing `(18, 19)` remap is doing exactly what Deep Research recommends doing for `(86, 87)`. **Do not drop the remap.** Strike Experiment 3 from the Cycle 1 list.

**Open item:** verify whether `Pushing_something_from_left_to_right` / `..._from_right_to_left` (SSv2 native `(93, 94)`) is present in our 33-class subset. User believes (18, 19) is the *only* mirror pair, but this needs `ls data/train/` confirmation. If a Pushing pair is present, add it to the remap.

### B. The biggest hidden constraint: data is 4 frames duplicated 4× into 16 slots

**What the doc says:** Nothing — the agent assumed the standard SSv2 input distribution (16 real frames per clip). All expected-gain estimates and the Experiment 1 ceiling (74–76 %) silently rely on this.

**What's actually true:** The professor's distribution gives **4 frames per clip**, sampled from the first 60 % of each SSv2 video. The dataloader duplicates each frame 4× to fill the 16-slot input tensor expected by the V-JEPA fpc16 architecture. So half the temporal token-position pairs see *zero motion delta*. The Kaggle test set is also in this distribution (otherwise scoring would be unfair).

**Implications for the plan:**

1. **Experiment 1 expected ceiling revised down: 74–76 % → ~71–73 %** with wider uncertainty. The SSv2-FT head Meta released was trained on real 16-frame sequences, so it has never seen the zero-motion-delta token structure we'll feed it. LoRA closes some of that gap but cannot recover Meta's full 73.7 % SSv2-174 ceiling.
2. **Full SSv2 download (user's step 2) is only useful if re-downsampled to the 4-frame regime first.** Pull each clip → take first 60 % of frames → subsample to 4 → duplicate 4× → 16-slot tensor. Mixing real 16-frame training data with 4-dup test data is a distribution-shift trap that will *hurt* test accuracy. Verify the professor's exact sampling stride against original SSv2 by comparing one clip's frames byte-for-byte.
3. **This is the novelty angle for the professor.** Frame: "Transferring a video model from a 16-frame pretraining distribution to a 4-frame-duplicated test distribution." The V-JEPA 2 paper does not address this regime, and it's a methodologically interesting question (low-bandwidth video inference, edge deployment). Both the *result* and the *finding-that-disabling-hflip-helped* are publishable observations independent of the leaderboard rank.

### C. Class-index mapping is required before Experiment 1 can run

**What the doc says:** §2 step 2 — "Identify the 33 target indices in the official SSv2 id2label space (which is the same label space as the HF model's `config.id2label`)." Caveat at §210 acknowledges this might not be free.

**What's actually true:** Our folders are renamed with a local prefix; the class-name string after the prefix matches SSv2's `id2label` (verified: `020_Putting_something_behind_something` matches SSv2's "Putting something behind something"). So the map is buildable, just not automatic.

**Concrete prerequisite for Day 1:** write a `local_to_ssv2_idx.py` helper that:
1. Lists `data/train/*/` folder names.
2. Strips the `NNN_` prefix and replaces underscores with spaces.
3. Looks each up in `VJEPA2ForVideoClassification.from_pretrained("facebook/vjepa2-vitl-fpc16-256-ssv2").config.id2label` (inverse map).
4. Returns a `(33,)` index tensor that selects 33 rows from the (174, hidden) classifier head.

Flag any folder name that doesn't resolve — there were precedents in the existing zero-shot script for a truncated folder name (class 015) that needed a token-aligned unique-prefix fallback.

### D. Hardware and parallelism — confirmed

- Currently on an A4000 16 GB. Will move to RTX 3090 24 GB before launching E1.
- "Parallel" experiments = multiple physical machines × 1 GPU each (not multi-GPU per host). Means r=8 vs r=16 and DoRA vs LoRA from Cycle 2 can genuinely run on different machines and produce comparable numbers, provided every machine is on the same git commit (which is why we did the merge first).
- Disk: 876 GB free, enough for the full-SSv2 download and re-downsampling pipeline.

### E. V-JEPA 2.1 — do not pursue

Decided. The +0.4 pp ViT-G gain is not worth the integration risk in our window, weights aren't on HF, and even hitting 1st or 2nd place is enough. Skip §8 of the body.

### F. Eval config recipe — caveat unresolved

§208 caveat: "exact contents of `configs/eval/vitl/ssv2.yaml` could not be retrieved verbatim". The config the user shared from Meta's repo is the V-JEPA 2.1 *pretraining* config, not the SSv2 evaluation config — so this caveat is still open. Before Cycle 2's 4-block-probe experiment (E4), pull the actual `vitl/ssv2.yaml` and the `src/datasets/` dataloader from `facebookresearch/vjepa2` to confirm hflip-off behavior, crop layout, and the multi-block-probe head architecture.

---

## User's plan, reconciled with the doc

The user laid out a 5-step plan in parallel with reading the Deep Research output. Mapping:

| User step | Maps to | Notes |
|---|---|---|
| 1. Try TTA variants on existing 68 % model | Experiment 2 (clean 2×3 no-flip protocol), evaluated on the *current* SSL-only checkpoint | Quick: tells us the upper bound of TTA-only gains without retraining. Useful sanity check. |
| 2. Download full SSv2, complete subset with missing clips | NOT in Deep Research; user addition | See §B above — only useful if re-downsampled to 4-frame format. Can run in background while E1 trains. |
| 3. Rerun the same recipe with no hflip on `vjepa2-vitl-fpc16-256-ssv2` | Experiment 1 (checkpoint swap) | The headline experiment. Revised expectation ≈ 71–73 %. |
| 4. Parallel r=8 vs r=16, then DoRA vs LoRA (for report) | Experiments 7 (LoRA rank/MLP target) and 8 (DoRA) | Run on different machines once `track-b` branch is everywhere. Ablation data for the report. |
| 5. Frozen `vjepa2-vitl-fpc16-256-ssv2` + attentive-probe baseline | Reproduces Meta's published 73.7 %, *but in our 4-frame regime* | This becomes the "what's the ceiling under temporal distribution shift" data point — directly answers the novelty question. Should be one of the first runs. |

### Novelty angles for the professor

The professor wants methodological novelty beyond "we used Meta's recipe". Any of the following qualifies:

- **4-frame temporal distribution shift on a 16-frame-trained head** (strongest; ties together every other finding).
- **LoRA + DoRA on `vjepa2-vitl-fpc16-256-ssv2`** — Meta has not published either combination on top of their own SSv2 finetune.
- **Direction-aware hflip on SSv2** — the +3.5 pp from disabling hflip is itself a publishable observation, since SSv2 papers tend to disable hflip by default without ablating it; we have the ablation.
- **Dataset reconstruction via subsampled full-SSv2 augmentation** under the same 4-frame distribution.

---

## What to do tomorrow, in order

### Before any training — preflight (≈ 2 hours total)

1. **Build the local → SSv2 native index map** (§C). Output: a saved `local_to_ssv2_idx.pt` tensor of shape `(33,)`. Fail loudly if any folder name doesn't resolve.
2. **Verify directional pair coverage** (§A). `ls data/train/ | grep -E "Pushing.*left|Pushing.*right"`. If both directions present, add `(local_push_lr_idx, local_push_rl_idx)` to the existing TTA remap.
3. **Confirm the dataloader's 4-dup-to-16 behavior** in `src/smth2smth/shared/data/video_dataset.py` — sample one clip end-to-end and inspect the tensor: positions 0,1 should equal 2,3 should equal 4,5 etc. if duplication is happening.
4. **Pull the SSv2-FT processor settings** from `facebook/vjepa2-vitl-fpc16-256-ssv2` — frame-sampling stride, normalization mean/std, image_size. Confirm `num_frames=16` is what the head expects.

### Calibration run (1 hour) — gate for the rest of Cycle 1

5. **Zero-shot eval** of `vjepa2-vitl-fpc16-256-ssv2` on `data/val/` with no training. Slice the 174-d head to our 33 indices using the map from step 1, feed 4-dup-to-16 frames, measure top-1 / top-5. Decision tree:
   - **≥ 70 %**: Deep Research expectation stands. E1 likely lands 74–76 %.
   - **55–65 %**: revised expectation (71–73 %) stands; proceed with E1 as planned.
   - **40–55 %**: temporal sensitivity is worse than expected. E1 ceiling probably 65–70 %; consider more LoRA epochs or revisit step 5 of the user plan (frozen-probe baseline) as the primary path.
   - **< 40 %**: something is wrong (wrong processor, frame-count mismatch, label-map bug). Debug before training.

### Experiment 1 (Days 1–2, depending on calibration)

6. **Launch the checkpoint-swap run.** Recipe from doc §2 + Experiment 1:
   - `VJEPA2ForVideoClassification.from_pretrained("facebook/vjepa2-vitl-fpc16-256-ssv2", num_labels=33, ignore_mismatched_sizes=True)` — then copy the sliced 33 rows from the original head before training.
   - LoRA r=16, α=32, targets `q_proj,k_proj,v_proj,o_proj` (start without MLP; add in Cycle 2 / Experiment 7 if Cycle 1 plateaus), dropout 0.05.
   - Head lr 5e-4, LoRA lr 1e-4, AdamW wd 0.05, cosine schedule, 15 epochs, bf16, no random hflip, batch 4 × 16 frames × 256².
   - TTA at submit: official 2 segments × 3 crops, **no flip**, mean-softmax. Keep the `(18, 19)` remap (and `(push_lr, push_rl)` if present).

### Background while E1 trains (Days 1–2)

7. **Start the full-SSv2 download** for the 33 classes present in our subset. Re-downsample each clip to 4 frames at the professor's sampling stride. Save to a parallel `data/train_extra/` directory; don't overwrite the existing `data/train/`.
8. **Run the frozen-probe baseline (user step 5)** on another machine: same checkpoint, no LoRA, only the head trains. Numbers go into the report as the "without our methodological contribution" baseline. This and E1 differ in exactly one variable (LoRA on / off), which is the ablation the professor will want.

### Cycle 2 gating (Day 3+)

If E1 + clean TTA lands at or above the calibration's predicted band, proceed to Cycle 2 (E4–E7) per the doc. If it lands meaningfully below, revisit assumptions before scaling up.

---

## Open verification items (in case any are wrong)

- The 4-dup-to-16 behavior is *believed* to be how the dataloader works based on user statement; step 3 of preflight is the actual verification.
- The folder-name → SSv2 class-name mapping is *believed* to be exact-string-match-after-prefix-stripping; step 1 of preflight will catch any deviations.
- The Pushing L↔R pair is *believed* not to be in the 33 classes; step 2 of preflight confirms.
- The expected band of the calibration eval (`≥ 70 %`, `55–65 %`, etc.) is based on rough intuition about how distribution shift affects temporally-trained heads. The actual number teaches us how to interpret all downstream results.

---

## TL;DR
- **The single highest-EV experiment is to swap the starting checkpoint from `facebook/vjepa2-vitl-fpc64-256` (SSL-only) to `facebook/vjepa2-vitl-fpc16-256-ssv2` (already supervised-finetuned on the full 174-class SSv2 by Meta FAIR) and re-use it as a 33-class head-sliced + LoRA-tuned probe at 16 frames.** That checkpoint hosts the same Meta-released SSv2 attentive probe that scores 73.7% top-1 on the full 174-class SSv2 val (Assran et al., V-JEPA 2, arXiv 2506.09985), so on a 33-class subset the head-slice ceiling is already at or above the 75% you are chasing.
- **Stop fighting TTA: the V-JEPA 2 official eval protocol is 2 segments × 3 spatial crops (6 views) with no flip; your 12-view (2×3×2 with flip) is double-counting direction-flipped views on a direction-sensitive dataset, which is the structural reason it regressed −9.6 pp.** Replace with the official 2×3 protocol and drop the flip TTA entirely.
- **All other levers (DoRA, higher LoRA rank, EMA, spatial-only VideoMix, multi-query/4-block probe, last-K-block token concat) are smaller +0.3 to +1.5 pp wins; do them in Cycle 2 only after locking in the checkpoint swap + clean TTA.** ViT-g and any train-time labeled-flip pipeline are anti-patterns for your VRAM / timeline.

## Key Findings

### 1. Ranked answer to "cheapest single win" (Q1)

| Option | Evidence basis | Expected pp | Effort |
|---|---|---|---|
| **(a) Swap to `vjepa2-vitl-fpc16-256-ssv2` + head-slice/LoRA** | Meta FAIR's own attentive-probe checkpoint scores 73.7% on full SSv2-174 val (Assran et al., V-JEPA 2 paper, arXiv 2506.09985, Tab. 4; HF model card). A 33-class subset is strictly easier. | **+4 to +7 pp** | ★ low |
| **(e) Fix the flip-pair remap TTA / use clean 2×3 protocol** | Official `configs/eval/vitg-384/ssv2.yaml` uses `num_segments: 2`, `num_views_per_segment: 3`. The user's (18,19) remap is NOT a directional flip pair — actual flip pairs from the verified SSv2 id2label are (86,87) "Pulling L↔R" and (93,94) "Pushing L↔R". | **+1 to +3 pp** (recovers part of the −9.6 pp loss) | ★ low |
| **(c) Higher LoRA rank (r=16 or 32, α=2r)** | Original LoRA paper (Hu et al., arXiv 2106.09685) treats higher rank as the lever when target capacity is insufficient; Unsloth LoRA-hyperparam guidance recommends r ≥ 16 and α=2r for new-task fitting. | +0.5 to +1.5 pp | ★ low |
| **(b) DoRA in place of LoRA** | Liu et al. ICML 2024 oral (arXiv 2402.09353): DoRA "consistently outperforms LoRA" on LLaMA/LLaVA/VL-BART; CLIP-DoRA (ScienceDirect S1877050925021787) reports "average improvements of up to 0.28% over the previous state-of-the-art" on 11 few-shot vision tasks — note that comparison is vs prior PEFT SOTA, not necessarily vanilla LoRA. Gains are real but small in the vision regime. | +0.2 to +0.8 pp | ★ medium |
| **(d) ViT-g** | Infeasible on 24 GB at 256² with 64 frames. Meta's own attentive-probe training at ViT-g/384 requires `nodes: 16`, `tasks_per_node: 8`, `mem_per_gpu: 220G` per the official config — a 128-GPU job. | n/a | ✗ blocked |

### 2. Best protocol for transferring `vjepa2-vitl-fpc16-256-ssv2` to a known 33-class subset (Q2)

Strong recommendation: **head-slice the 174-class linear head down to your 33 indices, calibrate, then add a small LoRA on top of the (still frozen) encoder.** Concretely:

1. Load `VJEPA2ForVideoClassification.from_pretrained("facebook/vjepa2-vitl-fpc16-256-ssv2")`.
2. Identify the 33 target indices in the official SSv2 id2label space (which is the same label space as the HF model's `config.id2label`).
3. Index the head weight/bias along dim 0 to get a (33, 1024) / (33,) head. This already gives a strong zero-shot result substantially above the 45% reported, because the 141 dropped classes can no longer steal probability mass.
4. Add `peft.LoraConfig(r=16, lora_alpha=32, dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj","mlp.fc1","mlp.fc2"])` to the encoder.
5. Train at **16 frames / clip** (the SSv2-FT head was trained at fpc16, not fpc64 — feeding 64 frames will mismatch the temporal positional/pooling distribution the head was trained on).
6. Optional: keep the 174-class head frozen and add a 174→33 mapping layer trained with knowledge distillation from the frozen 174 logits; in practice the simple slice is competitive and far cheaper.

**Do NOT re-initialize the head from scratch.** That discards Meta's already-trained class prototypes for the 33 target classes — empirically that loses several pp on similar partial-label transfer setups.

**fpc16 ↔ fpc64 mismatch implications:** the SSv2-FT model was trained with 16 frames × tubelet 2 = 8 temporal token steps. Using fpc64 with this checkpoint sends 32 token steps through positional/temporal embeddings the head never saw. Use 16 frames during fine-tuning with this checkpoint. If you want 32+ frames, stay on the `vjepa2-vitl-fpc64-256` SSL base and pay the larger transfer-learning cost.

### 3. V-JEPA 2 LoRA/PEFT recipes (Q3)

There is **no published Meta-blessed LoRA recipe specifically for V-JEPA 2**. The official evaluation protocol is **frozen encoder + 4-block attentive probe** (verified from `configs/eval/vitg-384/ssv2.yaml`: `num_probe_blocks: 4`, `num_heads: 16`), trained with a parallel-probe sweep over 5 LRs × 4 weight decays (LR ∈ {5e-3, 3e-3, 1e-3, 3e-4, 1e-4}; WD ∈ {0.01, 0.1, 0.4, 0.8}), `num_epochs: 20`, `warmup: 0.0`, `use_bfloat16: true`. Community fine-tunes (`SujitShelar/vjepa2-vitl-fpc16-256-hmdb51`, `qubvel-hf/vjepa2-vitl-fpc16-256-ssv2`, `eagle0504/vjepa2-vitl-fpc16-256-ssv2-ucf101`) all use either frozen-backbone + classifier head or backbone-frozen + gradient accumulation; none publish a LoRA configuration.

The closest blessed PEFT pattern for ViTs is the general Unsloth / community recommendation: target attention **and** MLP (`q,k,v,o,fc1,fc2`), r=16–32, α=2r, lr=1e-4 for adapter, 5e-4 for head, dropout 0.05. Attention-only LoRA is documented as a weaker configuration in recent LoRA-vs-FT studies on language models, and the analogous evidence on ViT image-classification (torchtune Llama2 LoRA tutorial demonstrates roughly +4 pp absolute improvement on truthfulqa_mc2 when extending LoRA from attention-only to all linear layers and increasing rank to 64). Apply this directionally.

### 4. SSv2 directional pairs and train-time labeled-flip (Q4)

From the official SSv2 id2label (verified verbatim on `huggingface/label-files`), the unambiguous horizontal-flip class pairs in the 174-class space are exactly:
- **86 ↔ 87**: "Pulling [something] from left to right" ↔ "Pulling [something] from right to left"
- **93 ↔ 94**: "Pushing [something] from left to right" ↔ "Pushing [something] from right to left"

Several other directional concepts in SSv2 — "Moving away from / towards the camera" (41/44), "Moving down / up" (43/45), "Moving away from / closer to" (40/42), "Approaching / Moving away with your camera" (0/32), and "Letting roll along/down/up a slanted surface" (22/23/24) — are NOT preserved under horizontal flip; they are vertical, depth, or context changes, so you cannot legally remap them under hflip. **Your current TTA remap of (18,19) ("Holding ... in front of" ↔ "Holding ... next to") is NOT a horizontal-flip pair** — those classes are not mirror images of each other, and remapping them likely introduces label noise.

**There is no published recipe for train-time labeled-flip-with-remap on SSv2.** The seminal evidence base for SSv2 + flip is negative: Price & Damen 2019 ("Retro-Actions", arXiv 1909.09422) explicitly note that "without temporal ordering, individual frames from a video clip of an 'open jar' action cannot be distinguished from frames of a 'close jar'", and treat flip-with-relabel as a zero-shot synthesis trick, not a standard augmentation. The standard published recipe (VideoMAE NeurIPS 2022 supplementary; V-JEPA 2 configs) is **flip disabled** during SSv2 training. Your +3.53 pp from disabling flip already captured the main win.

If you want to try labeled-flip-with-remap during training, the safe set is only {86, 87, 93, 94} AND you must verify those classes are present in your 33-class subset. For samples in those four classes, apply hflip with p=0.5 and swap the label to the partner; for all other classes keep hflip disabled. Expected gain ≤ +0.5 pp, with non-trivial regression risk if the pair is split (e.g., 86 is in the subset but 87 is not).

### 5. Multi-crop/multi-segment TTA "done right" (Q5)

The canonical V-JEPA 2 SSv2 evaluation protocol is **2 segments × 3 spatial views (6 views total), no flip**, codified in `configs/eval/vitg-384/ssv2.yaml` (`num_segments: 2`, `num_views_per_segment: 3`; checkpoint tag `ssv2-vitg16-384-64x2x3` and `ssv2-vitl-16x2x3.pt`). VideoMAE NeurIPS 2022 supplementary likewise uses **2 clips × 3 crops** on SSv2 (vs. 5×3 on Kinetics and 10×3 on HMDB51), explicitly because SSv2 is temporally constrained and multi-clip averaging breaks SSv2's temporal structure more easily than appearance-based datasets.

**Mechanism for your −9.6 pp regression** with 12 views (2 segments × 3 crops × 2 flips):
- Half of your 12 views are horizontally flipped, which on SSv2 systematically converts e.g. "Pulling L→R" features into "Pulling R→L" features. Logits are averaged across views, so direction-sensitive classes get pulled toward the wrong logit by 50% of the views — a structural failure mode, not a noise issue.
- The official 256² ViT-L probe was trained at 16×2×3 (filename of the official probe checkpoint: `ssv2-vitl-16x2x3.pt`); a heavier 64-frame multi-segment scheme at test time also shifts the distribution of temporal aggregation seen by the probe.

**Correct TTA for your setup**: 2 temporal segments × 3 spatial crops (center + 2 short-side crops), 16 frames per clip when using the fpc16 SSv2-FT checkpoint, **no flip**, average softmax (not logits) across views. Expected: recover most of the −9.6 pp and net ≈ +1 to +3 pp over single-view eval.

### 6. SSv2 SOTA 2025–2026 (Q6)

- **V-JEPA 2.1 ViT-G** (Mur-Labadia et al., arXiv 2603.14482, v1 submitted 15 Mar 2026): **77.7%** SSv2 — "V-JEPA 2.1 ViT-G achieves a top-1 accuracy of 77.7 on SSv2, setting a new absolute state-of-the-art on the task compared to 77.5 for InternVideo2 full fine-tuning, 77.3 for V-JEPA 2 ... and 69.7 for InternVideo2 using the same protocol" (paper HTML, arxiv 2603.14482v2).
- **V-JEPA 2 ViT-g/16 at 384** (Assran et al., 2025): 77.3% (paper-confirmed).
- **V-JEPA 2 ViT-L/16 at 256** (your backbone): **73.7%** with Meta's own 4-block attentive probe (README of facebookresearch/vjepa2).
- **InternVideo2 full FT**: 77.5%.
- **VideoMAE V2 ViT-g**: 77.0%.
- **InternVideo2 ViT-L (V-JEPA 2 protocol)**: 69.7%.

Papers With Code was sunsetted by Meta on 24 July 2025 (Julien Chaumond, HF CTO, public statement 25 July 2025), and its domain redirects to Hugging Face Trending Papers; there is no maintained public leaderboard surface for SSv2 in 2026. The numbers above are the most credible 2025–2026 results, all using the standard SSv2 val.

### 7. Kaggle / GitHub SSv2 write-ups (Q7)

No public Kaggle SSv2 competition writeups surfaced for the 33-class subset task. The closest analog is the Perception Test Challenge 2024 winning solution (Han et al., arXiv 2410.09088), which used VideoMAE v2 + UMT features and **explicitly augmented their training set with SSv2 samples for overlapping classes** — a useful pattern: if the official SSv2 train data for your 33 classes is accessible, include all of it (you already do "merge val into train" which gave +0.90 pp; pulling in the full SSv2 train rows for those 33 classes if not already present is a free win).

### 8. V-JEPA 2.1 availability (Q8)

V-JEPA 2.1 (Mur-Labadia et al., arXiv 2603.14482) was released 2026-03-16 with code only in `app/vjepa_2_1/` of the official repo; pretrained weights are **not on HuggingFace as of May 2026**. There is an open request issue (facebookresearch/vjepa2#137) and a pending community PR to integrate V-JEPA 2.1 into `transformers` (huggingface/transformers#45496). Currently weights are loadable only via Meta's torch.hub interface. ViT-L variant exists; ViT-G is 2 B params and won't fit your 24 GB. Reported SSv2 gain over V-JEPA 2: +0.4 pp at ViT-G scale (77.3 → 77.7); the ViT-L gain is unreported but Fig 5 of the paper suggests Deep Self-Supervision recovers classification accuracy at ViT-L. **Not worth the integration risk in a 10-day window** — defer to a follow-up.

### 9. Cheap regularization evidence base (Q9)

For your overfitting symptom (train 95% / val 69% by epoch 11), in priority order of published support specifically on video / frozen-probe / SSv2:

| Lever | Published evidence | Expected |
|---|---|---|
| **Multi-block (depth-4, 16-head) attentive probe — Meta's official architecture** | This is the probe that produces 73.7% on SSv2 ViT-L/256 per facebookresearch/vjepa2 README. Your single-query single-block probe is the structurally weaker variant. | +1 to +3 pp (probably the largest single probe-side win) |
| **EMA (decay 0.999–0.9999) of probe/LoRA params** | Morales-Brotons et al. 2024 (arXiv 2411.18704): "a consistent improvement in generalization using an EMA model" across image classification tasks, including in frozen-feature linear-probe transfer (e.g., TinyImageNet→CIFAR-100 linear probe 52.77% → 57.78%). PyTorch torchvision issue #4346: "Most of SOTA models use EMA to get a few extra accuracy points for free." | +0.5 to +1.5 pp |
| **VideoMix (Yun et al. 2020, arXiv 2012.03457) — spatial CutMix only, no temporal cuts** | VideoMix paper shows accuracy gains on K400/SSv2 after 200 epochs vs baseline; Vi-Mix (ICLR 2022, OpenReview 00Vc1Ov5KZn) shows that **temporal** CutMix HURTS SSv2 because "cutmix operation destroys the temporal structure of the videos which is crucial for understanding actions in videos" — restrict to spatial cuts identical across all 16 frames. | +0.5 to +1.0 pp |
| **Layer-wise LR decay on LoRA (0.85^layer top→bottom)** | Standard transfer-learning practice (towardsdatascience.com layer-wise LR decay write-ups); cheap. Limited direct video evidence. | +0.2 to +0.5 pp |
| **Last-K-block token concat for the probe** | DINOv2 and PE-Core (Bolya et al. 2025) evaluations use last-block-set concat for frozen-backbone probing; routinely +0.3–0.8 pp. | +0.3 to +0.8 pp |
| **More input frames (24/32 vs 16) on fpc64 base model** | The fpc64 SSL base accepts up to 64 frames; VideoMAE supplementary reports ViT-L SSv2 32×1×3 at 75.4 vs 16×2×3 at 74.3. Helps only if you stay on the fpc64 SSL base; if you switch to the SSv2-FT checkpoint you are pinned to 16 frames. | +0.3 to +1.0 pp (fpc64 base only) |
| **Gradient clipping (max-norm 1.0)** | Standard; no downside. | +0.0 to +0.2 pp |
| **SAM/ASAM** | 2× train time; not justified at your timeline. | +0.5 pp at ~2× cost |
| **Feature-level mixup** | Limited published video evidence; speculative. | unknown |

## Details — Ranked shortlist of next experiments

### Cycle 1 (Days 1–4): Checkpoint swap + clean TTA — must-do

**EXPERIMENT 1: Swap to `facebook/vjepa2-vitl-fpc16-256-ssv2`, slice head to 33, freeze backbone, train head + LoRA at 16 frames.**
- Expected: 68.81% → **74–76%** (within striking distance of 1st place at 75%).
- Effort: 4–6 engineering hours (recipe swap + head slicing + processor change to a 16-frame sampler). Training wall-clock: ~12–18 h on a single RTX 3090 for 15–20 epochs at batch 4, 16 frames, 256². Use AMP bf16 (Meta's config uses `use_bfloat16: true`) rather than fp16.
- Reproduce: `VJEPA2ForVideoClassification.from_pretrained("facebook/vjepa2-vitl-fpc16-256-ssv2", num_labels=33, ignore_mismatched_sizes=True)` — then manually copy the 33 sliced rows from the original `classifier.weight`/`classifier.bias` before training. LoRA: r=16, α=32, target `q_proj,k_proj,v_proj,o_proj`, dropout 0.05, LoRA lr 1e-4, head lr 5e-4, AdamW wd 0.05, cosine schedule, 15 epochs, no random hflip.

**EXPERIMENT 2: Replace your 12-view TTA with the official 2 segments × 3 crops, no flip.**
- Expected: recover most of your −9.6 pp loss → net **+2 to +3 pp** over current best.
- Effort: 1–2 engineering hours; ≤ 1 h eval wall-clock.
- Reproduce: Sample 2 temporal segments (start-half + end-half), apply 3 spatial crops (top-left/center/bottom-right for landscape or left/center/right on short side at 256²), no hflip, mean softmax. Mirrors `num_segments: 2, num_views_per_segment: 3` in the official `configs/eval/vitl/ssv2.yaml`.

**EXPERIMENT 3: Drop the (18,19) class-pair remap from TTA.**
- Expected: +0.2 to +0.5 pp. Classes 18 ("Holding ... in front of") and 19 ("Holding ... next to") are not mirror images of each other.
- Effort: 5 minutes.

### Cycle 2 (Days 5–8): Probe architecture and regularization

**EXPERIMENT 4: Replace the single-query, single-block attentive probe with a 4-block, 16-head attentive classifier (Meta's official architecture).**
- Expected: +1 to +3 pp. Structurally the largest probe-side change.
- Effort: 4–6 engineering hours (port the `AttentiveClassifier(embed_dim=1024, num_heads=16, depth=4, num_classes=33)` from `facebookresearch/vjepa2/notebooks/vjepa2_demo.ipynb`). Training wall-clock: ~12 h.
- Hyperparameters: Use a small slice of Meta's parallel-probe sweep: lr ∈ {1e-3, 3e-4, 1e-4}, wd ∈ {0.01, 0.1}, warmup 0.

**EXPERIMENT 5: Add EMA (decay 0.9998) over probe + LoRA params; evaluate the EMA model.**
- Expected: +0.5 to +1.5 pp on val. Strong fit for your overfitting profile.
- Effort: 1 engineering hour (`torch.optim.swa_utils.AveragedModel` with a custom `avg_fn` for EMA). VRAM cost: a second copy of trainable params only (~12 MB for r=16 LoRA + 4-block probe), trivial on a 3090.

**EXPERIMENT 6: Add spatial-only VideoMix (CutMix applied identically across all 16 frames, α=1.0, p=0.5) + keep label smoothing 0.1.**
- Expected: +0.5 to +1.0 pp. Yun et al. arXiv 2012.03457; restrict to spatial cuts to avoid SSv2 motion-structure degradation reported by Vi-Mix (ICLR 2022, OpenReview 00Vc1Ov5KZn).
- Effort: 2 engineering hours.

**EXPERIMENT 7: Bump LoRA rank to r=16 with α=32 (or r=32 / α=64) and extend target modules to `q_proj,k_proj,v_proj,o_proj,mlp.fc1,mlp.fc2`.**
- Expected: +0.3 to +1.0 pp over r=8 attention-only. torchtune's Llama-2 LoRA tutorial reports ~+4 pp on truthfulqa_mc2 from this exact change at r=64 with all linear layers; even partial application should help.
- Effort: 0.5 engineering hours; training wall-clock similar.

### Cycle 3 (Days 9–10): Polish / optional

**EXPERIMENT 8: DoRA in place of LoRA (PEFT 0.10+ supports DoRA natively with `use_dora=True`).**
- Expected: +0.2 to +0.8 pp over equivalent-rank LoRA. Liu et al. ICML 2024 oral arXiv 2402.09353; treat the upper end as optimistic for vision (CLIP-DoRA reports only +0.28% avg vs prior PEFT SOTA on 11 few-shot vision tasks).
- Effort: 5 minutes config change; same train wall-clock.

**EXPERIMENT 9: Direction-aware train-time hflip on the safe pair set only.**
- IF your 33-class subset contains {86, 87, 93, 94}: apply hflip with p=0.5 to all samples; for samples in those four classes, swap the label to the partner. Otherwise keep hflip disabled.
- Expected: +0.2 to +0.5 pp; non-zero regression risk.
- Effort: 2 engineering hours.

**EXPERIMENT 10: Last-2-block token concatenation as input to the attentive probe.**
- Expected: +0.3 to +0.8 pp; common DINOv2 / PE-Core trick.
- Effort: 2 engineering hours.

## Anti-patterns (do not do)

**Backbone / scale:**
- ViT-g at any resolution on a single 24 GB 3090 — infeasible. Meta evaluates at 16 nodes × 220 GB/GPU.
- V-JEPA 2.1 — weights not on HF as of May 2026; integration risk dominates expected gain in a 10-day window.

**TTA / TTA-shape mismatches:**
- 12-view TTA with horizontal flip on SSv2 — direction-encoded labels pull average logits in the wrong direction.
- Single-clip eval — leaves +1 to +3 pp on the table.
- 32-frame eval on the fpc16 SSv2-FT checkpoint — temporal pos-embed mismatch.

**Augmentation:**
- Train-time random hflip without class-pair remap on SSv2 (you already saw +3.53 pp from removing this).
- Temporal CutMix / temporal slice mixing (destroys motion structure on SSv2 per Vi-Mix ICLR 2022).
- Remap of (18,19) — not a mirror pair.
- Random rotation, ColorJitter with hue > 0.1 — no published SSv2 benefit and they amplify the train/val gap.

**Training tricks:**
- Full fine-tune of the ViT-L encoder on 33-class label budget — well under 332 M-parameter capacity; expect strong overfitting (your r=8 LoRA already overfits to train 95%).
- Knowledge distillation from a teacher you cannot evaluate independently.
- Repeated low-LR "continuation" runs without an EMA or a held-out gate — these can drift toward the seen subset.

**Held-out-test anti-patterns (relevant because your eval is a held-out Kaggle split, NOT a live leaderboard):**
- Pseudo-labeling on the public test set or any iterative self-training using your own predictions on Kaggle test — overfits the public test distribution and will not generalize to the held-out split.
- Heavy hyperparameter sweeps optimized against the Kaggle public LB rather than against your local val.
- Multi-checkpoint ensembling tuned by public LB score — same overfitting risk; ensemble only if all members are selected on local val.
- Per-class threshold/temperature tuning fit on public LB.

## Recommendations (concrete, with thresholds)

1. **Day 1–2 (Cycle 1a):** Run Experiment 1 (checkpoint swap) + Experiment 2 (clean 2×3 no-flip TTA) + Experiment 3 (drop bad (18,19) remap). **Threshold to continue:** if local val top-1 ≥ 73.5%, proceed to Cycle 2 as planned. If < 71%, suspect a frame-count or processor mismatch with the fpc16 checkpoint (32 frames being fed to a 16-frame head, wrong processor, etc.) and debug before training more.
2. **Day 3–4 (Cycle 1b):** Train two seeds of the best Cycle 1 config to estimate noise (typical ±0.3 pp), then lock in the configuration.
3. **Day 5–7 (Cycle 2):** Run Experiments 4 (4-block probe), 5 (EMA), 6 (spatial VideoMix), 7 (LoRA r=16, MLP target) — stack them on the Cycle 1 winner one at a time, keeping each that improves local val by > 0.3 pp. **Threshold:** if any single change regresses by > 0.5 pp on local val, revert it; do not chase the public LB.
4. **Day 8–10 (Cycle 3 + freeze):** Experiment 8 (DoRA) if spare time. Final submission is the EMA model of the best Cycle 2 config, evaluated with the official 2×3 protocol and no flip. Keep 2 alternate checkpoints (Cycle 1 winner, Cycle 2 winner) as ensemble candidates; only ensemble if cross-validated local val confirms a gain.
5. **Decision threshold for V-JEPA 2.1:** if Meta uploads `vjepa2.1-vitl-*` to HF before Day 5 AND the SSv2 attentive-probe weights ship with it, swap; otherwise skip.

### Completion check

| Spec item | Covered |
|---|---|
| Ranked 5–10 experiments with effort + pp + repro | ✓ (10 experiments, hours + wall-clock + hyperparams) |
| Cheapest single win ranking (a–e) | ✓ (table in §1) |
| fpc16-256-ssv2 → 33-class subset protocol | ✓ (§2: head-slice + LoRA at 16 frames) |
| V-JEPA 2 LoRA recipes | ✓ (§3) |
| SSv2 directional pairs | ✓ (§4: (86,87), (93,94)) |
| TTA done right + 12-view regression mechanism | ✓ (§5) |
| SSv2 SOTA 2025–2026 | ✓ (§6) |
| Kaggle / GitHub writeups | ✓ (§7) |
| V-JEPA 2.1 availability | ✓ (§8: not on HF) |
| Regularization evidence | ✓ (§9 table) |
| Anti-patterns | ✓ (dedicated section) |
| Public-LB-overfitting flagged | ✓ (held-out-test subsection) |
| 24 GB VRAM ceiling accounted | ✓ (§1 ViT-g, EMA cost note) |
| 10-day weekday timeline / 2–3 cycles | ✓ (Cycle 1/2/3 plan) |

## Caveats

- The 73.7% SSv2 number for V-JEPA 2 ViT-L/16 at 256 is on the full 174-class val with Meta's official 4-block attentive probe; your single-query single-block probe is structurally weaker, which is part of why your 68.81% is below ceiling.
- The exact contents of `configs/eval/vitl/ssv2.yaml` could not be retrieved verbatim during this research; inference is from the structurally identical `vitg-384/ssv2.yaml` (which was retrieved verbatim) and the published checkpoint filename `ssv2-vitl-16x2x3.pt`. Train/val augmentations live in the dataloader code (`src/datasets/`), not in YAML — confirm hflip behavior in the eval code if you intend to mirror Meta's recipe exactly.
- Public LB at 68.81% is a noisy signal: with a held-out Kaggle test split, the gap to local val matters more than the gap to public LB. Track local val first; treat LB as a sanity check, not a target.
- "33-class subset" is assumed to be 33 of the 174 official SSv2 classes with the same id-space and same train videos; if your subset uses re-indexed class IDs or non-overlapping data, the head-slice trick still works but requires building the id map yourself.
- DoRA's reported gains are modest in the vision regime (CLIP-DoRA: "average improvements of up to 0.28% over the previous state-of-the-art" on 11 few-shot vision tasks, ScienceDirect S1877050925021787). The +0.7 pp figure sometimes cited for DoRA vs LoRA comes from the original DoRA paper's language/VL benchmarks, not vision tasks — treat the vision-side expectation as the lower end of the published range.
- The V-JEPA 2 paper SSv2 numbers are attentive-probe with frozen encoder; full-fine-tune numbers (e.g., InternVideo2 77.5% full-FT) are not directly comparable and are not your operating regime on a 24 GB 3090.
- The LoRAT paper (arXiv 2403.05231) is a tracking paper, not video classification; r=64 on ViT as a default originates from the original LoRA paper (Hu et al., arXiv 2106.09685) — your r=16/32 recommendation is conservative relative to that baseline.