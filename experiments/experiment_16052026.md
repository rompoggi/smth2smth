# Track A — Overnight Experiment Batch (2026-05-16)

**Goal:** maximize Kaggle **Top-1** accuracy on the closed-world action-anticipation task.
**Format:** 7 independent experiments. Each runs on **one** GPU/VM (RTX 3090, 24 GB) **overnight**.
**Hard rule:** the VMs **cannot communicate**. Every experiment below is a *complete pipeline*
(pretrain → fine-tune, or full supervised train) that depends on **no** artifact produced by
another experiment. No experiment may point `model.init_from` at another VM's encoder.

This file is a **specification only** — no code, no launch. Implementation is a later step.

---

## 0. Shared constraints (apply to all 7 — do not violate)

- **Closed world.** No external data, no pretrained weights, no third-party checkpoints.
SSL pretraining is allowed **only** on the provided `train/ + val/ + test/` frames
(labels not consumed).
- **Hardware.** One RTX 3090, 24 GB. Every experiment ships an explicit batch size,
AMP dtype, and grad-accum so it fits in 24 GB.
- **Honest val (mandatory for every reported number).**
`dataset.use_official_val=true` **and** `dataset.include_val_in_train=false`.
Any run that puts val labels into training is **leaky** and its val number is void
(this already burned us once — see `experiments.md`, rouget Phase-2b: ~60% leaky val,
only 36% public).
- **VideoMAE multi-scale TTA is patch-bound.** `PatchEmbed3D` requires spatial sides
divisible by `patch_size=16`. At base 224 the **only** legal triple is
**{192, 224, 256}** → `tta_scales=[0.857, 1.0, 1.143]`.
`[0.875, 1.0, 1.125]` (→196, 252) **crashes** the ViT and must never be used on a
VideoMAE model. The R50+TSM CNN is fully convolutional and may use `[0.875, 1.0, 1.125]`.
- **Submit-time hygiene (every model, before any Kaggle upload):**
  1. confirm the `[tta] flip-pair remap: 018<->019, ...` line lists ≥1 pair (else flip
    produces wrong labels for direction-sensitive classes);
  2. confirm `len(checkpoint["extra"]["trained_class_indices"])` matches the 50-class
    head and is **not** the legacy 33.
- **Unique outputs.** Each experiment uses its own codename → its own log file and its own
`checkpoints/track_a/ssl/<codename>_*.pt` so that, if results are gathered later onto one
host, nothing collides.
- **Monitoring.** Rely on the ISO-8601 timestamped epoch lines (SSL.md §6): pretrain
`[videomae] <ts> epoch k/N avg loss ...`; FT `[<ts>] Epoch k/N | ...`. A run with no
new timestamp for >1 h is dead — do not wait on it.

### Not part of this batch

- The **Phase-2 R50+TSM 150-epoch** run is *already alive on another host* and is the
current best single-model path (~42% public, lost-ckpt rerun in progress). It is **not**
one of these 7 and must not be relaunched here. **E4** below is a *distinct* CNN variant
(snapshot ensemble), not a duplicate of it.
- **Rejected — do not run on any VM:** mHC / Hyper-Connections (LLM-scale fix, no evidence
<100 M params, high CUDA risk); PixMix (needs external fractal data — illegal);
train-on-val (leaky); ViT-from-scratch without SSL (6 logs ≤6%, verdict closed);
cross-VM knowledge distillation (needs a teacher checkpoint from another VM — violates
the independence rule).

### "Champion FT stack" (referenced by several experiments)

The best-evidence VideoMAE fine-tune recipe, assembled from Tier-S/A of the research docs:


| Knob                                                | Value                   | Source                                 |
| --------------------------------------------------- | ----------------------- | -------------------------------------- |
| `model.variant`                                     | `vit_s`                 | baseline                               |
| `model.head` / `head_num_heads`                     | `attn` / 4              | baseline                               |
| `model.drop_path_rate`                              | **0.2**                 | A2 (VideoMAE supp Tab 6)               |
| `model.dropout` (head)                              | **0.5**                 | A2 (Phase-2 CNN parity)                |
| `training.optimizer`                                | `adamw`                 | baseline                               |
| `training.lr` (encoder peak)                        | **5e-4**                | S1 (raised for LLRD)                   |
| `training.layer_decay`                              | **0.75**                | **S1** (VideoMAE official FINETUNE.md) |
| `training.weight_decay`                             | 0.05                    | baseline                               |
| `training.epochs`                                   | **60**                  | A3                                     |
| `training.warmup_epochs`                            | 6                       | A3                                     |
| `training.min_lr`                                   | 1e-6                    | A3                                     |
| `training.scheduler_cosine`                         | true                    | baseline                               |
| `training.early_stopping_patience`                  | 25                      | A3                                     |
| `training.label_smoothing`                          | 0.1                     | baseline                               |
| `training.amp` / `amp_dtype`                        | true / **bf16**         | baseline                               |
| `training.ema_enabled` / `ema_decay`                | true / **0.9999**       | baseline                               |
| `training.videomix_mode`                            | **mixup_cutmix_switch** | **S2**                                 |
| `training.videomix_mixup_alpha`                     | 0.8                     | S2                                     |
| `training.videomix_cutmix_alpha`                    | 1.0                     | S2                                     |
| `training.videomix_prob` / `switch_prob`            | 1.0 / 0.5               | S2                                     |
| `training.repeated_aug`                             | **2**                   | **S5**                                 |
| `augment.preset`                                    | `randaugment_t`         | A4                                     |
| `augment.randaug_n` / `randaug_m`                   | **2 / 9**               | A4 (VideoMAE FT recipe)                |
| `augment.randaug_mode`                              | `temporal_plus`         | A4                                     |
| `augment.random_horizontal_flip` / `flip_prob`      | **true / 0.5**          | **A1** (+ submit L/R remap)            |
| `augment.random_crop_pad`                           | 32                      | baseline                               |
| `dataset.time_reversal_prob`                        | 0.0                     | rejected on leading path               |
| `dataset.use_official_val` / `include_val_in_train` | true / false            | honest val                             |


When repeated_aug=2 is on, halve clips/step (`training.batch_size=8`) so the effective
view-batch stays 16; VRAM ≈ 14 GB bf16 ViT-S.

---

## E1 — SSL champion (the flagship single SSL model)

- **Codename:** `requin`
- **Hypothesis:** the full Tier-S/A FT stack on a standard VideoMAE encoder is our best
single closed-world SSL model and beats the 35.6% piranha baseline by +1.5–3 pp honest.
- **Pipeline (one VM, full):**
  - **Phase 1 — MAE pretrain.** ViT-S, `pretrain.mask_ratio=0.75`, tube mask,
  `pretrain.epochs=100`, `pretrain.lr=1.5e-4`, `pretrain.warmup_epochs=5`,
  `pretrain.weight_decay=0.05`, `pretrain.betas=[0.9,0.95]`, `pretrain.batch_size=16`,
  bf16. **Minimal** spatial aug only: `random_crop_pad=32`, no flip, no jitter
  (this is the *control* pretrain; E2 is the augmented-pretrain contrast).
  → `checkpoints/track_a/ssl/requin_encoder.pt`.
  - **Phase 2 — FT.** Champion FT stack (table above), `init_from=requin_encoder.pt`.
  → `requin_ft.pt` (best EMA).
- **Budget:** ~7 h pretrain + ~4.5 h FT ≈ **11.5 h**. Fits one night.
- **Submit config (ViT):** `tta=true tta_flip=true tta_scales=[0.857,1.0,1.143]`,
softmax-mean aggregation, `tta_logit_adjust=0.0`.
- **Success criterion:** honest EMA val Top-1 **≥ 37.0%** by ep 60 (+1.4 pp vs piranha).
- **Failure handling:** if train loss never drops below ~2.5 by ep 20, the stack is
over-regularised — note it and let it run (do not hand-tune overnight); E6 is the
lighter-reg fallback.

## E2 — SSL with augmented MAE pretrain (resolves a doc disagreement)

- **Codename:** `murene2`
- **Hypothesis & why it matters:** the two research docs **disagree**. Ranked-recipes **A5**
says add `RandomHorizontalFlip(0.5)` + light `ColorJitter(0.4,0.4,0.4,0.1)` +
`RandomGrayscale(0.2)` to MAE pretrain (flip is label-free; reconstruction is invariant).
The pipeline-improvements doc §A says MAE aug must stay **strictly minimal** or the
decoder reconstructs augmentation noise. This experiment settles it.
- **Pipeline (one VM, full):**
  - **Phase 1 — MAE pretrain, augmented.** Same as E1 Phase 1 **except**:
  `random_horizontal_flip=true`, `flip_prob=0.5`,
  `color_jitter=(0.4,0.4,0.4,0.1)`, `color_jitter_prob=0.8`,
  `random_grayscale_prob=0.2`; **no RandAugment-T at pretrain**;
  `pretrain.epochs=150` (longer to amortise harder pretext). → `murene2_encoder.pt`.
  - **Phase 2 — FT.** Champion FT stack, `init_from=murene2_encoder.pt`.
- **Budget:** ~8–9 h pretrain (150 ep) + ~4.5 h FT ≈ **13–14 h**. Tight overnight — start early.
- **Submit config:** ViT triple `[0.857,1.0,1.143]` + flip.
- **Success criterion:** matched-FT honest EMA val **≥ E1 + 0.5 pp**. If it is **lower**
than E1, the minimalist-MAE camp is right and A5 is rejected for our regime — that is a
valid, useful result.
- **Comparison anchor:** E1 (`requin`) is the control. Both use the identical champion FT,
so the only variable is the pretrain augmentation.

## E3 — Temporal capacity: T=8 frames end-to-end

- **Codename:** `congre`
- **Hypothesis:** doubling temporal context (T=4→8) at *both* pretrain and Ft is worth
+0.5–1.5 pp despite ~2× per-epoch cost; action *anticipation* is temporally starved at T=4.
- **Pipeline (one VM, full):**
  - **Phase 1 — MAE pretrain, T=8.** ViT-S, `dataset.num_frames=8`, `mask_ratio=0.75`
  tube, `pretrain.epochs=80` (reduced for the longer clips), bf16,
  `pretrain.batch_size=8` + `grad_accum_steps=2` + gradient checkpointing.
  → `congre_encoder.pt`.
  - **Phase 2 — FT, T=8.** Champion FT stack with `dataset.num_frames=8`,
  `training.batch_size=4` + `grad_accum_steps=2` + gradient checkpointing,
  `training.epochs=50`. `init_from=congre_encoder.pt`.
- **Critical constraint:** T must be **8 at pretrain, FT, and submit** — never change T
between train and inference (breaks ViT pos-embeddings / TSM shift). Submit with
`dataset.num_frames=8`.
- **Budget:** ~8 h pretrain (80 ep, slow clips) + ~5 h FT ≈ **13 h**. Tight — start early.
- **Submit config:** ViT triple `[0.857,1.0,1.143]` + flip, T=8.
- **Success criterion:** honest EMA val **≥ E1 + 0.5 pp** at T=8. If equal/worse, T=4 is
confirmed sufficient and this lever is closed.

## E4 — CNN snapshot ensemble (R50+TSM, SGDR), distinct from the live 150-ep run

- **Codename:** `tsm_sgdr`
- **Hypothesis:** a single SGDR run with cosine warm restarts yields ensemble-grade
predictions at single-run cost (Snapshot Ensembles), giving a strong **CNN** submission
that is architecturally orthogonal to the SSL ViT runs (good independent error modes).
- **Pipeline (one VM, full, from scratch — closed world):**
  - `avanced_resnet50_tsm` (TSM in every bottleneck conv1, shift_div=8), from scratch
  (`model.init_from=null`), T=4, image 224, official val.
  - `training.scheduler=sgdr`, `sgdr_cycles=3`, **30 epochs/cycle = 90 epochs total**,
  `lr_max=1e-4`, `lr_min=1e-6`, AdamW `wd=0.05`, AMP fp16, EMA 0.999,
  attentive head 4h, `drop_path_rate=0.1`, `dropout=0.5`, `label_smoothing=0.1`,
  `videomix_mode=frame_mixup` α=5.0 p=0.5, RandAugment-T (n=2,m=9,temporal_plus),
  `random_horizontal_flip=true`, `random_crop_pad=32`.
  - **Save one snapshot at the bottom of each of the 3 cosine cycles**
  (`*_snap1.pt`, `*_snap2.pt`, `*_snap3.pt`) plus the EMA best.
- **Budget:** 90 epochs ≈ overnight if ≤ ~9 min/epoch; **if epoch time pushes total > 14 h,
reduce to `sgdr_cycles=3 × 24 ep = 72`** (note this in the log). Do not exceed the night.
- **Submit config (CNN):** softmax-average the 3 snapshots, then `tta=true tta_flip=true tta_scales=[0.875,1.0,1.125]` (CNN-safe), softmax-mean.
- **Success criterion:** snapshot-averaged honest EMA val **≥ best single snapshot + 0.5 pp**,
and competitive with the dual-stream A1b 39.47% honest baseline.
- **Note:** independent of the live 150-ep Phase-2 run — different schedule, different host,
different checkpoints. Both can coexist; this one tests the snapshot-ensemble hypothesis.

## E5 — SSL champion + decoupled long-tail recalibration (cRT)

- **Codename:** `silure2`
- **Hypothesis:** instance-balanced features + a short class-balanced **classifier-only**
retrain (Decoupling / cRT) fixes tail-class decision boundaries without corrupting the
MAE-learned backbone — cheaper and safer than class-balanced loss on the whole net.
- **Pipeline (one VM, full):**
  - **Phase 1 — MAE pretrain.** Identical to E1 Phase 1 (control pretrain, 100 ep,
  minimal aug). → `silure2_encoder.pt`.
  - **Phase 2a — FT stage 1 (representation).** Champion FT stack, **50 epochs**,
  instance-balanced (uniform) sampling, `init_from=silure2_encoder.pt`.
  - **Phase 2b — FT stage 2 (classifier).** Freeze entire backbone + attentive pool;
  retrain **only the classifier** for **10 epochs** with class-balanced
  (`sqrt_inverse`) sampling, `lr=1e-4`, cosine to 1e-6. → `silure2_ft.pt`.
- **Budget:** ~7 h pretrain + ~4 h stage1 + ~0.7 h stage2 ≈ **12 h**.
- **Submit config:** ViT triple `[0.857,1.0,1.143]` + flip; also report a
`tta_logit_adjust ∈ {0.0,0.5,1.0}` sweep **on honest val only** (A7) and submit the τ
that wins val by ≥0.2 pp, else τ=0.
- **Success criterion:** honest EMA val **≥ E1 + 0.3 pp**, with per-class recall on the
bottom-10 classes improved vs E1 (the actual point of cRT).
- **Comparison anchor:** E1 (`requin`) — same pretrain + same stage-1 FT, so the only
variable is the stage-2 classifier recalibration.

## E6 — SSL champion at fine-tune resolution 256 (train/test-res fix + clean lighter reg)

- **Codename:** `dorade`
- **Hypothesis:** (a) fine-tuning at 256 fixes the train/test resolution discrepancy for a
near-free boost, and 256 is patch-divisible (256/16=16); (b) doubles as the
lighter-regularisation fallback if E1's heavy stack over-regularises.
- **Pipeline (one VM, full):**
  - **Phase 1 — MAE pretrain @224.** Identical to E1 Phase 1 (100 ep, minimal aug).
  → `dorade_encoder.pt`.
  - **Phase 2 — FT @256.** Champion FT stack **except**: `dataset.image_size=256`,
  `model.interpolate_pos_embed=true`, `training.batch_size=12` (VRAM at 256),
  `**model.drop_path_rate=0.1` and `model.dropout=0.0`** (lighter reg — this is the
  deliberate contrast to E1's 0.2/0.5), `training.epochs=50`.
  `init_from=dorade_encoder.pt`.
- **Budget:** ~7 h pretrain + ~5 h FT@256 ≈ **12 h**.
- **Submit config:** base 256; ViT patch-divisible scales relative to 256 →
use `tta_scales=[0.875,1.0,1.125]`? **No** — recompute: at base 256 the divisible
triple is {224, 256, 288} → `tta_scales=[0.875,1.0,1.125]` (224/256=0.875, 288/256=1.125;
224,256,288 all ÷16). Confirm sides at submit start. Flip on, softmax-mean.
- **Success criterion:** honest EMA val **≥ E1 + 0.5 pp** (resolution + lighter-reg combined).
Also tells us, vs E1, whether E1's 0.2/0.5 reg was net positive at 60 ep.

## E7 — MAE cosine mask schedule (0.90 → 0.75) ablation

- **Codename:** `lamproie`
- **Hypothesis:** decaying the mask ratio from 0.90 to 0.75 over pretraining forces global
semantics early and local detail late, preventing early representation collapse on a
<100 k-clip set — beating a static-0.75 pretrain on downstream FT.
- **Pipeline (one VM, full):**
  - **Phase 1 — MAE pretrain, scheduled mask.** ViT-S, **cosine mask schedule
  `mask_ratio: 0.90 → 0.75`** over `pretrain.epochs=150`, tube mask, minimal aug
  (as E1 Phase 1 otherwise), bf16, bs 16. → `lamproie_encoder.pt`.
  - **Phase 2 — FT.** Champion FT stack, `init_from=lamproie_encoder.pt`.
- **Budget:** ~8–9 h pretrain (150 ep) + ~4.5 h FT ≈ **13–14 h**. Start early.
- **Submit config:** ViT triple `[0.857,1.0,1.143]` + flip.
- **Success criterion:** honest EMA val **≥ E1 + 0.5 pp** (E1 = static-0.75 control with the
identical champion FT). If ≤ E1, static 0.75 is confirmed and mask scheduling is closed.
- **Note:** do **not** test static 0.85 — both docs agree 0.85 starves the model at T=4;
the cosine schedule is the only defensible mask-variant to spend a VM on.

---

## Summary table


| ID  | Codename | What it tests                     | Pretrain                     | FT                           | ~Wall  | Beats if             |
| --- | -------- | --------------------------------- | ---------------------------- | ---------------------------- | ------ | -------------------- |
| E1  | requin   | Champion SSL stack (control)      | MAE 100ep, min-aug           | champion 60ep                | 11.5h  | ≥37.0% honest        |
| E2  | murene2  | Augmented vs minimal MAE pretrain | MAE 150ep, +flip/jitter/gray | champion 60ep                | 13–14h | ≥ E1 + 0.5pp         |
| E3  | congre   | T=8 temporal capacity             | MAE 80ep T=8                 | champion 50ep T=8            | 13h    | ≥ E1 + 0.5pp         |
| E4  | tsm_sgdr | CNN SGDR snapshot ensemble        | — (from scratch)             | SGDR 3×30ep                  | ≤14h   | ≥ best snap + 0.5pp  |
| E5  | silure2  | Decoupled long-tail (cRT)         | MAE 100ep, min-aug           | champion 50ep + 10ep cls     | 12h    | ≥ E1 + 0.3pp & tail↑ |
| E6  | dorade   | FT @256 + lighter reg             | MAE 100ep @224               | champion 50ep @256 dp0.1/do0 | 12h    | ≥ E1 + 0.5pp         |
| E7  | lamproie | Cosine mask schedule 0.90→0.75    | MAE 150ep, sched mask        | champion 60ep                | 13–14h | ≥ E1 + 0.5pp         |


**Cross-experiment design:** E1 is the shared control. E2/E5/E6/E7 each change exactly one
factor vs E1 (pretrain aug / cRT stage / resolution+reg / mask schedule), so any win is
cleanly attributable. E3 (T=8) and E4 (CNN) are orthogonal architecture/temporal bets and
also yield ensemble members with independent error modes for a later softmax-average
submission. No experiment reads another's checkpoint — fully VM-independent.

**After the batch (next step, not tonight):** softmax-average the best ViT SSL model with
the live Phase-2 R50+TSM CNN at submit time (per-model TTA first). That inference-only
ensemble is the single highest-leverage move and needs no further training.