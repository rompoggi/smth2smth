# Diffusion and generative frame densification for Track B (4 → 16 frames)

**Status:** exploratory proposal — **not** used in the current training run (Run 9).  
**Baseline in production:** temporal **duplication** via `pick_frame_indices(4, 16)` (see [Run 9 config](../configs/experiment/track_b_vjepa2_hfclf_16f_lora_r16.yaml)).  
**Sanity-check artifacts:** `logs/frame_densify_sanity_4to16/`.

---

## 1. Problem we are trying to solve

### 1.1 Data regime (professor distribution)

- Each clip on disk has **exactly four** RGB frames, sampled from the **first 60%** of the original Something-Something v2 video.
- The Kaggle / course test set uses the **same** distribution (fair evaluation).
- Meta’s Track B backbone `facebook/vjepa2-vitl-fpc16-256-ssv2` was **fine-tuned with 16 frames per clip** (tubelet size 2 → **8 temporal tokens** in the encoder).

So we face a **structural mismatch**:

| Side | Temporal input |
|------|----------------|
| Meta SSv2-FT weights | 16 **distinct** (or smoothly varying) frames |
| Our folders | 4 **sparse** keyframes |

### 1.2 Two ways to fill 16 slots

| Method | Mechanism | Used today? |
|--------|-----------|-------------|
| **A. Duplicate** | `pick_frame_indices(4, 16)` — linspace over anchor indices with repeats, e.g. `[0,0,0, 1,1,1,1,1, 2,2,2,2,2, 3,3,3]` | **Yes** (Run 9, all prior V-JEPA Track B runs with `T=16` or `T=8`) |
| **B. Synthesize** | Insert **12** intermediate frames (4 per gap between consecutive anchors) so the clip **looks** like a 16-frame video | **No** (only offline sanity check) |

**Diffusion / generative interpolation** is a subclass of **B**: learn a model that predicts pixels (or latents) between `I_k` and `I_{k+1}` instead of repeating anchors.

Target timeline (uniform spacing, anchors fixed):

```text
I₁ — f₁ — f₂ — f₃ — f₄ — I₂ — f₅ — f₆ — f₇ — f₈ — I₃ — f₉ — f₁₀ — f₁₁ — f₁₂ — I₄
     └──────── 4 mids ────────┘     └──────── 4 mids ────────┘     └──────── 4 mids ────────┘
```

Total: **4 anchors + 12 synthesized = 16 frames**.

---

## 2. Why consider diffusion or generative VFI at all?

### 2.1 Limitation of duplication (Run 9)

Duplication gives the **correct tensor shape** for fpc16 but **not** correct motion semantics:

- Many consecutive frames are **identical** → zero optical flow between tubelet pairs.
- The attentive pooler and LoRA adapters must infer action from **jumps** between repeated blocks, not from smooth trajectories.
- This is the main reason we **do not** expect to match Meta’s published ~73.7% SSv2-174 ceiling even with the same checkpoint and LoRA recipe (see `track_b_next_steps.md`, annotation B).

### 2.2 Limitation of classical interpolation (already tested)

We ran an offline sanity check with **OpenCV Farneback flow + warp** (not diffusion), script `scripts/sanity_check_frame_densify_4to16.py`, module `src/smth2smth/shared/data/frame_densify.py`.

**Observed failure mode (qualitative):** on fast hand motion, the hand **appears abruptly** or **smears** rather than moving continuously — typical of **large displacement + occlusion** between sparse anchors. Flow-based methods **warp existing pixels**; they cannot invent a hand that was outside the field of view in `I_k` but visible in `I_{k+1}`.

**Generative / diffusion-style models** are attractive precisely because they **inpaint disoccluded regions** under a learned prior, instead of only blending warped textures.

### 2.3 What “diffusion” means in this document

We use **diffusion** loosely to cover:

1. **True diffusion VFI** — video latent denoising between endpoints (e.g. LDMVFI, some DynamiCrafter variants).
2. **Generative VFI** — flow + refinement networks trained with reconstruction / perceptual losses (MoG-VFI, etc.).
3. **Strong learned interpolators** — not diffusion in the DDPM sense, but often compared in the same bucket (FILM, RIFE, TLB-VFI).

The **design question** for smth2smth is the same: *can we synthesize 12 believable in-betweens per clip without hurting fine-grained verb classification?*

---

## 3. What could be good (potential benefits)

1. **Temporal alignment with Meta’s fpc16 weights**  
   Sixteen **non-identical** frames may reduce the “zero-motion delta” pathology and use the SSv2-FT temporal positional embeddings as intended.

2. **Smoother motion cues for fine-grained verbs**  
   Classes that depend on **how** something moves (pulling L↔R, pretending vs real put, pick vs put) might gain signal if mids show **continuous** hand/object trajectories instead of frozen repeats.  
   See error analysis: `docs/track_b_vjepa2_lora_r16_confusion_matrix_analysis.md` (pretend/real pairs, class 22 “put into” sink).

3. **Methodological story**  
   “Sparse keyframes → densified clip → V-JEPA” is a clear pipeline for the report, especially if contrasted with duplication ablation on the **same** LoRA recipe.

4. **Offline preprocessing**  
   Densification can run **once** on disk (new folder layout or cache); training loop stays unchanged except `num_frames` ingestion path. No need to run diffusion inside the training step.

5. **Tiered cost–quality**  
   Not every clip requires the heaviest model; bisection with a mid-tier model (FILM/RIFE) may suffice for small motion gaps.

---

## 4. What could be bad (risks and failure modes)

### 4.1 Hallucination and label corruption

| Risk | Why it matters for SSv2 |
|------|-------------------------|
| **Wrong hand pose / finger count** | Pretend verbs (e.g. class **016**) differ from real actions by **intent**, not by gross motion template. |
| **Object appearance drift** | “Put into” vs “pretend put” — model may invent container interactions (confusion with class **022**; 202 false preds to 022 on full val in CM analysis). |
| **Texture prior from pretraining** | Diffusion models bias toward “plausible video,” not “true continuation of this clip.” |
| **Left/right flip ambiguity** | Even with `random_horizontal_flip: false`, synthesized frames might blur L/R motion (class **018** pulling pair). |

**Key point:** For **classification**, a slightly blurry **true** frame can be better than a sharp **wrong** frame.

### 4.2 Distribution shift vs train and test

- If we densify **train** but not **test** (or vice versa), accuracy collapses.
- If we densify with a model trained on **natural 30 fps video** but apply it to **4 sparse SSv2 keyframes** (large inter-frame gap), quality may be poor without domain-specific fine-tuning.
- Must **not** mix **native 16-frame SSv2** training data with **4-frame densified** data unless test is densified the same way (`track_b_next_steps.md`).

### 4.3 Compute and engineering

- Full-dataset densification: ~45k train + 6.7k val clips × 3 gaps × several forward passes per gap (recursive bisection).
- Diffusion per pair is **orders of magnitude** slower than duplication; storage doubles or triples if we write 16 JPEGs per clip.
- Reproducibility: stochastic samplers → need fixed seeds and version-pinned weights for ablations.

### 4.4 No free lunch on occlusions

If the hand is **not visible** in `I_k` but **is** in `I_{k+1}`, **any** method must guess the intermediate state. Diffusion guesses **more convincingly** — which may **increase** classifier confidence on the wrong verb.

### 4.5 Evidence from our flow sanity check

We **rejected** flow warp for production after visual inspection. Generative methods may fix pulling/picking clips but **amplify** pretend/put confusion if they “complete” the action too eagerly.

**Reference comparison (same clip, two rows):**

| Artifact | Path |
|----------|------|
| Side-by-side grid | `logs/frame_densify_sanity_4to16/018_Pulling_something_from_left_to_right__video_101343_compare.png` |
| Duplicate playback | `.../018_Pulling_something_from_left_to_right__video_101343_duplicate_playback.mp4` |
| Flow-interp playback | `.../018_Pulling_something_from_left_to_right__video_101343_flow_playback.mp4` |

**Test clip (primary):**

- **Class:** `018_Pulling_something_from_left_to_right`  
- **Video id:** `video_101343`  
- **On-disk path:** `data/val/018_Pulling_something_from_left_to_right/video_101343/` (4 frames)  
- **Why this clip:** large horizontal hand motion between anchors; flow warp failed visibly (“hand pops in”). Any generative proposal should be judged here **before** scaling.

**Secondary test clip (fine-grained verb stress):**

- `data/val/016_Pretending_to_put_something_into_something/video_103485/`  
- Compare: `logs/frame_densify_sanity_4to16/016_Pretending_to_put_something_into_something__video_103485_compare.png`  
- Use to check **hallucinated contact** with container / hand.

---

## 5. Architectures we could use (recommended tiers)

Do **not** start with full text-to-video diffusion. Prefer **endpoint-conditioned** models (two images in → one or more frames out).

### Tier 0 — Baseline (current Run 9)

- **Duplicate:** `pick_frame_indices(4, 16)` in `VideoFrameDataset`  
- **Cost:** zero offline  
- **Role:** ablation lower bound / Kaggle-matched ingestion  

### Tier 1 — Learned flow VFI (not diffusion, try first if revisiting synthesis)

| Model | Role | Notes |
|-------|------|--------|
| **[FILM](https://github.com/google-research/frame-interpolation)** | Large-motion **single** mid-frame; recursive bisection for 4 mids/gap | Designed for wide baselines between photos; TensorFlow SavedModel |
| **[Practical-RIFE](https://github.com/hzwer/Practical-RIFE)** (v4.25) | Fast PyTorch; arbitrary `ratio` between two frames | Good throughput for dataset-wide preprocess |
| **TLB-VFI** | Strong on benchmarks; multi-frame output per call | Heavier integration |

**Pipeline:** per gap `(I_k, I_{k+1})`, recursive `t=0.5` splits until 4 mids; **never** single-jump `t ∈ {0.2,0.4,0.6,0.8}` on large gaps.

### Tier 2 — Generative VFI (light diffusion / flow+gen hybrid)

| Model | Role | Notes |
|-------|------|--------|
| **MoG-VFI** | Flow trajectory + generative correction | Needs per-pair or per-clip prompts in some setups |
| **LDMVFI** | Latent diffusion between frames | Better disocclusions; slower; monitor hallucination |

Use when Tier 1 still shows occlusion pops on the **018 / 101343** test clip.

### Tier 3 — Video diffusion interpolators (heavy)

| Model | Role | Notes |
|-------|------|--------|
| **DynamiCrafter (interp mode)** | Many frames between two conditioning images | ~16 frames per gap possible; text conditioning awkward for unlabeled SSv2 |
| **ToonCrafter / SEINE** | Long interpolation | Overkill; domain gap on real handheld SSv2 |

**Recommendation:** treat Tier 3 as **research optional**, not default preprocessing.

### Architecture choice summary

```text
                    ┌─────────────────────────────────────┐
  4 on-disk frames  │  Offline densify (chosen tier)      │
        │           │  per gap: I_k → 4 mids → I_{k+1}   │
        └──────────►│  → 16 JPEGs / clip on disk          │
                    └─────────────────┬───────────────────┘
                                      ▼
                    ┌─────────────────────────────────────┐
                    │  V-JEPA2 ViT-L fpc16 + LoRA r=16    │
                    │  (same as Run 9, new data path)     │
                    └─────────────────────────────────────┘
```

---

## 6. Proposed experimental protocol (before any full-scale run)

1. **Visual gate (2 clips)**  
   - `018` / `video_101343` (motion)  
   - `016` / `video_103485` (pretend / contact)  
   Regenerate with candidate model; compare to duplicate row in sanity-check layout.

2. **Short classification probe (~100 holdout clips)**  
   - Freeze Run 9 best checkpoint (or train 1–2 epochs on densified subset only).  
   - Metric: official val top-1 **and** per-class recall on 016, 018, 022, 011.

3. **Full preprocess + Train Run 10** only if step 2 does not regress holdout and pretend/put errors do not worsen in CM.

4. **Submission** must use **identical** densification on test folders.

---

## 7. Relation to current Run 9

| Question | Answer |
|----------|--------|
| Does Run 9 use diffusion? | **No.** |
| What does Run 9 use? | `dataset.num_frames: 16` + linspace **duplicate** of 4 anchors. |
| Log | `logs/track_b_vjepa2_hfclf_16f_lora_r16_20260523_025751.log` |
| W&B | `smth2smth-track-b` / `vitl_fpc16ssv2_16f_lora_r16` |

Diffusion densification would be a **separate** experiment (e.g. Run 10): new on-disk clips or on-the-fly loader branch `ingest: densified_vfi`, same Hydra preset otherwise.

---

## 8. Decision matrix (concise)

| Criterion | Duplicate (Run 9) | Learned VFI (FILM/RIFE) | Diffusion / generative VFI |
|-----------|-------------------|-------------------------|----------------------------|
| Fidelity to true motion | Low (piecewise constant) | Medium | Medium–high |
| Occlusion handling | N/A (repeats pixels) | Poor–medium | Good |
| Hallucination risk | Low | Low–medium | **High** |
| SSv2 fine-grained verbs | Safe but weak motion signal | Mixed | **Risky** |
| Compute | None | Moderate | High |
| Kaggle / test parity | Exact professor protocol | Needs same pipeline on test | Same |

**Practical conclusion for the project today:**

- **Run 9** correctly tests “does **T=16 alignment** + LoRA help when we still **duplicate**?”  
- **Diffusion / generative densification** remains a **hypothesis** worth documenting and testing on **`018/video_101343`** and **`016/video_103485`**, not a silent change to the training pipeline.  
- Escalate architecture tier only until those two clips pass visual inspection **and** a small holdout probe does not hurt pretend/put metrics.

---

## 9. References in this repo

| Item | Location |
|------|----------|
| Duplicate vs flow sanity script | `scripts/sanity_check_frame_densify_4to16.py` |
| Densify utilities | `src/smth2smth/shared/data/frame_densify.py` |
| Run 9 experiment | `configs/experiment/track_b_vjepa2_hfclf_16f_lora_r16.yaml` |
| Confusion / error analysis | `docs/track_b_vjepa2_lora_r16_confusion_matrix_analysis.md` |
| Distribution-shift context | `track_b_next_steps.md` (§B) |
| Track B run logbook | `report/track_b.tex` (Run 9) |
