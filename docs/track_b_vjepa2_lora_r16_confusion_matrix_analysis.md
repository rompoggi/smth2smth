# Error Structure on Official Validation for Track B V-JEPA2 LoRA (8f)

**Model:** `facebook/vjepa2-vitl-fpc16-256-ssv2` + LoRA \(r=16, \(\alpha=32\) on encoder attention + trainable Meta pooler + 33-row sliced head  
**Checkpoint:** `checkpoints/track_b/vitl_fpc16ssv2_8f_lora_r16.pt` (Run 7 best, epoch 2)  
**Eval split:** full official `data/val` (6,745 clips, 32 present classes)  
**Protocol:** center-crop eval transforms, bf16 AMP, **no TTA**, class index 27 masked at inference (never trained)  
**Date of analysis:** 2026-05-23  

**Artifacts:** `logs/track_b_vjepa2_hfclf_8f_lora_r16_fullval_cm/` (counts CSV, labeled/recall-normalized PNGs, `per_class_error_analysis.csv`, `meta.json`)

---

## Abstract

We evaluate the Run 7 Track B classifier at **70.64%** top-1 accuracy on 6,745 official validation clips and analyse the **1,980** misclassifications via a full \(33\times33\) confusion matrix. Errors are dominated by **semantic neighbour collisions**—especially *pretend* versus *real* verbs, upward-motion verbs (*pick* / *move up* / *pretend pick*), and a systematic **over-prediction of “put into”** (class 22)—rather than by extreme **frequency imbalance** in the training set. Training accuracy during Run 7 exceeded **99%**, indicating severe **generalization gap** on fine-grained motion semantics. We quantify the largest confusion pairs, decompose errors by verb family, discuss the missing class 27, and list mitigation strategies available in the `smth2smth` codebase (class-balanced sampling/loss, logit adjustment, short classifier refinement).

---

## 1. Experimental setup

### 1.1 Training regime (Run 7)

| Symbol | Meaning | Value |
|--------|---------|-------|
| `T` | Frames per clip | 8 (4 real frames duplicated in time) |
| `Sz` | Spatial size | 256 |
| `bs` | Batch size | 8, `grad_accum_steps=3` |
| `lr` | Head + LoRA LR | \(10^{-4}\), cosine + 1 warmup epoch |
| LoRA | Rank / \(\alpha\) | 16 / 32 on encoder `query,key,value,proj` |
| Aug | Policy | `vjepa2_heavy`, **horizontal flip off** (SSv2 directionality) |
| Val | Split | `use_official_val=true`, train = `data/train` only |
| Best val | Reported in log | **70.60%** top-1 (epoch 2) |

Resume (epochs 14–15) did not beat this checkpoint on full official val.

### 1.2 Evaluation for this analysis

- **Checkpoint weights** loaded from `vitl_fpc16ssv2_8f_lora_r16.pt` (not the val90 holdout classifier-tuned variant).
- **All** val folders under `data/val/` indexed; stratified holdout (10%) is **not** used here—this is the same metric as Run 7 training logs.
- **Untrained-class mask:** index **27** logits set to \(-\infty\) (0 train / 0 val clips; head slice 32/33).
- **No test-time augmentation** (single forward per clip); holdout-FT logs used `val_eval_tta=true` but `tta_scales=null` → effectively similar.

---

## 2. Global results

| Metric | Value |
|--------|-------|
| Correct | 4,765 |
| Errors | 1,980 |
| **Top-1 accuracy** | **70.64%** |
| Top-5 (not recomputed here) | — |
| Classes in val | 32 (of 33 configured) |
| Train clips (Run 7) | 44,993 |
| Train top-1 (end of training) | \(\approx\) 99% |

The **29 percentage-point** train–val gap is the central diagnostic: the model is not failing because it lacks capacity on the training distribution; it **collapses distinct val verbs into a small set of high-prior labels**.

---

## 3. Error taxonomy

We partition off-diagonal mass \(E=1980\) as follows:

| Category | Count | % of errors | Interpretation |
|----------|------:|------------:|----------------|
| Wrongly predicted as **22** (*put into*) | 202 | 10.2% | “Insert / transfer” sink class |
| **Pretend** true → **real** pred | 224 | 11.3% | Drops “pretending” cue |
| **Real** true → **pretend** pred | 207 | 10.5% | Inflated pretend labels |
| **Same verb family** (e.g. put↔put, move↔move) | 344 | 17.4% | Fine-grained within family |
| **Cross-family** (remaining) | 1,205 | 60.9% | Heterogeneous motion confusion |

**Note:** “Pretend ↔ pretend” cross-errors are rare as a *named* family because pretend classes mainly confuse with **real** counterparts, not each other.

### 3.1 Hypothesis classes of failure

1. **Visual synonymy:** Short clips with similar hand-object trajectories (upward reach, insertion, rotation).
2. **Pretend–real ambiguity:** SSv2 “pretending” classes differ by intent, not by stark motion profile; with **8 frames** and duplicated temporal sampling, subtle cues vanish.
3. **Label smoothing (0.1):** Encourages soft boundaries exactly where hard boundaries matter (pretend vs real).
4. **No class rebalancing:** `class_balance_sampler=none`, `class_balance_loss=none` → head prior follows raw sampling, not val-hardness.
5. **Overfitting:** Near-perfect train accuracy with frozen encoder + modest LoRA → pooler/head fits train co-occurrence statistics.

---

## 4. Systematic biases: what the model predicts when wrong

**Column analysis** (total predictions assigned to class \(p\) when true label \(\neq p\)):

| Rank | Class \(p\) | Label (short) | Times predicted (wrong) |
|-----:|------------:|-----------------|------------------------:|
| 1 | 22 | Putting into something | **202** |
| 2 | 11 | Picking something up | 168 |
| 3 | 14 | Pretending to pick up | 116 |
| 4 | 30 | Turning upside down | 108 |
| 5 | 24 | Putting onto something | 102 |
| 6 | 9 | Moving something up | 101 |
| 7 | 29 | Throwing something | 94 |
| 8 | 5 | Holding something | 84 |
| 9 | 31 | Uncovering something | 80 |
| 10 | 10 | Opening something | 80 |

**Interpretation:** The classifier behaves as if the label space were low-dimensional: **insert**, **lift**, **rotate**, **open**, **throw**. Rare or subtle verbs are absorbed into these modes.

---

## 5. Dominant confusion pairs (bidirectional)

Sorted by total off-diagonal flow \(C_{tp}+C_{pt}\):

| Total | Classes | \(t\to p\) | \(p\to t\) | Mechanism |
|------:|---------|----------:|----------:|-----------|
| **76** | 11 ↔ 14 | 37 | 39 | Pick ↔ pretend pick |
| **75** | 16 ↔ 22 | 54 | 21 | Pretend put ↔ put into |
| **71** | 9 ↔ 11 | 36 | 35 | Move up ↔ pick up |
| **61** | 2 ↔ 22 | 53 | 8 | Drop into ↔ put into |
| **45** | 17 ↔ 29 | 20 | 25 | Pretend throw ↔ throw |
| **40** | 3 ↔ 32 | 24 | 16 | Fold ↔ unfold |
| **40** | 9 ↔ 14 | 35 | 5 | Move up ↔ pretend pick |
| **33** | 11 ↔ 30 | 15 | 18 | Pick ↔ turn upside down |
| **32** | 8 ↔ 24 | 31 | 1 | Move down ↔ put onto |
| **31** | 10 ↔ 28 | 23 | 8 | Open ↔ take out |
| **28** | 0 ↔ 10 | 21 | 7 | Close ↔ open |
| **28** | 22 ↔ 24 | 12 | 16 | Put into ↔ put onto |
| **27** | 5 ↔ 25 | 7 | 20 | Hold ↔ show to camera |
| **26** | 25 ↔ 29 | 25 | 1 | Show ↔ throw |
| **23** | 9 ↔ 30 | 13 | 10 | Move up ↔ upside down |

**Counterfactual:** Fixing only the **top three pairs** perfectly would recover **222** clips → top-1 \(\approx\) **73.9%** (+3.3 pp). Most lift is concentrated, not diffuse.

### 5.1 Top 15 directed confusions (single cells)

| Count | True → Predicted |
|------:|------------------|
| 54 | Pretending to put into something → **Putting into something** |
| 53 | Dropping into something → **Putting into something** |
| 39 | Pretending to pick up → **Picking up** |
| 37 | Picking up → **Pretending to pick up** |
| 36 | Moving up → **Picking up** |
| 35 | Picking up → **Moving up** |
| 35 | Moving up → **Pretending to pick up** |
| 31 | Moving down → **Putting onto** |
| 25 | Throwing → **Pretending to throw** |
| 25 | Showing to camera → **Throwing** |
| 24 | Folding → **Unfolding** |
| 23 | Opening → **Taking out** |
| 21 | Putting into → **Pretending to put** |
| 21 | Closing → **Opening** |
| 20 | Showing to camera → **Holding** |

---

## 6. Per-class diagnosis (worst recall)

Full table in **Appendix A**. Below: classes with recall **< 65%** or special role.

### 6.1 Class 16 — *Pretending to put something into something* (recall **7.4%**)

| Stat | Value |
|------|------:|
| Val clips | 68 |
| Train clips | 1,044 |
| Correct | 5 |

**What goes wrong:** In **54/63** errors, the model predicts **real** “put into” (22). Additional confusion with drop-into (4) and move-down (2).

**Why:** Motion template = hand moves object toward container opening; “pretending” is weakly expressed in 8f duplicated clips. The head learned on train to map this template to **22** (high prior, many real put examples). This is the single worst class despite **mid-high** train frequency—not a long-tail artefact.

**What would help:** Pretend-specific negatives (16 vs 22) in loss; higher sampling weight on 16; lower label smoothing in a refinement phase; more temporal frames.

---

### 6.2 Class 11 — *Picking something up* (recall **32.7%**)

| Stat | Value |
|------|------:|
| Val clips | 199 |
| Train clips | 980 |
| Top confusions | 37→14 pretend pick, 35→9 move up, 15→30 upside down |

**What goes wrong:** Symmetric war with **pretend pick** (76 bidirectional errors combined with 14) and **move up** (71 with 9). The model cannot stabilize “committed lift” vs “incomplete / upward translation.”

**Why:** Upward hand trajectory is shared across pick, pretend-pick, move-up, and sometimes turn-upside-down. Train accuracy is high because templates separate on train co-occurrence; val clips expose overlap.

---

### 6.3 Class 17 — *Pretending to throw* (recall **38.3%**)

| Val / train | 47 / 915 |
|-------------|----------|
| Main error | **20** → class 29 *Throwing* |

Arm swing without release looks like throw; model defaults to real throw (also **25** throw→pretend throw in reverse).

---

### 6.4 Class 14 — *Pretending to pick something up* (recall **51.8%**)

| Val / train | 228 / 1,547 |
|-------------|-------------|
| Main error | **39** → class 11 *Picking* |

Mirror of §6.2; pretend pick is mistaken for real pick more than the converse on val (39 vs 37), but pair is nearly symmetric.

---

### 6.5 Class 24 — *Putting something onto something* (recall **52.5%**)

| Main errors | 16→22 put **into**, 8→cover, 7→behind |

Confusion between **onto** vs **into** vs **behind**—fine spatial relation errors; 22 again acts as attractor for insertion-like motion.

---

### 6.6 Class 2 — *Dropping something into something* (recall **57.9%**)

| Main error | **53** → 22 *Put into* |

**Drop** vs **place** share approach-to-container motion; release phase may be missed in short clips.

---

### 6.7 Class 5 — *Holding something* (recall **58.9%**)

| Main errors | 18→pick, 12→upside down, 10→pull R→L |

Static hold confused with onset of pick or rotation.

---

### 6.8 Class 25 — *Showing something to the camera* (recall **59.0%**)

| Main errors | 25→throw, 20→hold, 16→move up |

Presentation gesture overlaps throw wind-up and hold.

---

### 6.9 Class 26 — *Spilling something next to something* (recall **55.0%**, **rarest train count 162**)

| Main error | **19** → put next to |

Rare class + pour-like motion → neighbour relation errors. Here **frequency** may matter alongside semantics.

---

### 6.10 Strong classes (recall > 90%)

| Class | Name | Recall |
|------:|------|-------:|
| 18 | Pull L→R | **92.3%** |
| 19 | Pull R→L | 86.4% |
| 12 | Pour into | 86.7% |

Directional pull pair remains strong (hflip off + only mirror pair in subset). Pour-into is relatively isolated semantically.

---

## 7. Frequency imbalance vs semantic imbalance

### 7.1 Train distribution (44,993 clips)

| Class | Train % | Val clips | Recall |
|------:|--------:|----------:|-------:|
| 26 | 0.36% | 60 | 55.0% |
| 13, 15 | 0.70% | 79, 56 | 70.9%, 60.7% |
| 16 | 2.32% | 68 | **7.4%** |
| 9 | 7.05% | 359 | 65.5% |

**Conclusion:** Low recall does **not** correlate cleanly with low train %. Class **16** has 1,044 train examples and still fails catastrophically.

### 7.2 Train vs val count ratio per class

Per-class train/val clip ratio spans ~2.7×–19× (expected: similar val fraction per class). No missing val class except **27**.

---

## 8. Missing class 27

- Folder **027_** absent from train and val (head slice: 32/33 SSv2 ids resolved).
- Inference masks index 27 to \(-\infty\).
- Does not enter confusion matrix; Kaggle submissions never predict 27.

---

## 9. Comparison with val90 holdout classifier FT (context)

| Checkpoint | Eval set | Top-1 |
|------------|----------|------:|
| `vitl_fpc16ssv2_8f_lora_r16.pt` | 10% holdout only | 70.86% |
| `vitl_fpc16ssv2_8f_lora_r16.pt` | **full val** | **70.64%** |
| `vitl_fpc16ssv2_8f_lora_r16_val90_holdout_clf.pt` | 10% holdout | 74.11% |

Holdout FT (classifier-only on 90% val, LoRA frozen) improves **holdout** metrics but is **not** the checkpoint analysed in Sections 3–6. Full-val CM for the FT checkpoint would be needed to see if pretend-put (16) improved on all 6,745 clips.

---

## 10. Mitigation strategies (actionable)

### 10.1 Without architecture change (Hydra / eval)

```yaml
# Training continuation from vitl_fpc16ssv2_8f_lora_r16.pt
training.class_balance_sampler: sqrt_inverse   # or inverse
training.class_balance_loss: sqrt_inverse
training.class_balance_beta: 0.999
training.label_smoothing: 0.02                   # down from 0.1 for refinement
```

```bash
# Inference calibration (Menon logit adjustment)
training.tta_logit_adjust=0.5   # sweep on val
```

### 10.2 Short FT recipes (in-repo patterns)

1. **Val90 holdout + classifier-only** (already run): helps calibration on seen distribution; +3.5 pp on holdout vs base.
2. **Class-balanced LoRA FT** (5–10 epochs, `lr=5e-5`): targets put-sink and pick/move-up collisions.
3. **`val_eval_tta=true`** with scale sweep for reporting (may shift open/close, spatial classes).

### 10.3 Research directions (not implemented)

- **Pretend–real contrastive pairs** in batch construction (14/11, 16/22, 17/29).
- **More real temporal frames** (16f native) per `track_b_next_steps.md` §B.
- **Auxiliary “is_pretend” head** on pooler features.
- **Post-hoc pair-wise calibration** only for confused pairs (high risk on test).

---

## 11. Summary

| Question | Finding |
|----------|---------|
| What failed? | Fine-grained verb discrimination on val, not training convergence. |
| Worst classes? | **16** pretend put (7%), **11** pick (33%), **17** pretend throw (38%), **14** pretend pick (52%). |
| Main mixtures? | Pretend↔real; pick↔move-up↔pretend-pick; **put-into sink (22)**; open/close/take-out; fold/unfold. |
| Imbalance type? | Primarily **semantic / prior / overfitting**, secondarily frequency for rare spill/pour. |
| Quick wins? | Class-balanced sampler+loss; logit adjust; short FT; lower label smoothing. |
| Ceiling from pair fixes? | Top-3 pairs → ~**74%** val top-1 (upper bound if pairs only). |

---

## Appendix A — Per-class metrics (full val)

| Idx | Class (folder suffix) | Train \(n\) | Val \(n\) | Recall | Top confusions (count → class) |
|----:|------------------------|------------:|----------:|-------:|--------------------------------|
| 0 | Closing something | 1068 | 228 | 73.2% | Opening (21), put into (7), fold (7) |
| 1 | Covering with something | 2727 | 417 | 80.6% | uncover (9), put into (9), show (8) |
| 2 | Dropping into something | 903 | 178 | 57.9% | **put into (53)**, throw (4), behind (4) |
| 3 | Folding something | 972 | 285 | 82.5% | unfold (24), uncover (10), open (3) |
| 4 | Hitting with something | 1738 | 235 | 60.9% | onto (15), into (11), throw (7) |
| 5 | Holding something | 1459 | 197 | 58.9% | pick (18), upside down (12), pull RL (10) |
| 6 | Moving away | 910 | 183 | 73.2% | pull RL (9), pull LR (8), closer (7) |
| 7 | Moving closer | 907 | 213 | 85.4% | next to (7), pull RL (4), hit (4) |
| 8 | Moving down | 2741 | 311 | 68.8% | onto (31), hold (16), throw (12) |
| 9 | Moving up | 3170 | 359 | 65.5% | pick (36), pretend pick (35), upside down (13) |
| 10 | Opening something | 1253 | 332 | 73.2% | take out (23), upside down (17), unfold (7) |
| 11 | Picking up | 980 | 199 | **32.7%** | pretend pick (37), move up (35), upside down (15) |
| 12 | Pouring into | 873 | 278 | 86.7% | pour out (12), pretend pour (7), into (6) |
| 13 | Pouring out | 314 | 79 | 70.9% | pour in (11), pretend pour (7) |
| 14 | Pretend pick | 1547 | 228 | 51.8% | pick (39), pull RL (11), upside down (8) |
| 15 | Pretend pour (trunc.) | 314 | 56 | 60.7% | pour in (8), upside down (5), pour out (5) |
| 16 | Pretend put into | 1044 | 68 | **7.4%** | **put into (54)**, drop into (4) |
| 17 | Pretend throw | 915 | 47 | 38.3% | throw (20), hold (3) |
| 18 | Pull L→R | 1555 | 169 | 92.3% | pick (4) |
| 19 | Pull R→L | 1587 | 125 | 86.4% | uncover (4) |
| 20 | Put behind | 1204 | 127 | 73.2% | next to (8), into (5) |
| 21 | Put in front | 837 | 135 | 74.8% | cover (13), onto (5) |
| 22 | Put into | 2188 | 292 | 67.8% | pretend put (21), onto (12), take out (11) |
| 23 | Put next to | 2031 | 203 | 71.4% | closer (8), onto (7) |
| 24 | Put onto | 1608 | 139 | 52.5% | into (16), cover (8), behind (7) |
| 25 | Show to camera | 709 | 261 | 59.0% | throw (25), hold (20), move up (16) |
| 26 | Spill next to | 162 | 60 | 55.0% | next to (19), pour in (4) |
| 28 | Take out | 1699 | 239 | 82.8% | into (8), open (8), pick (5) |
| 29 | Throwing | 2254 | 184 | 74.5% | pretend throw (25), hold (4) |
| 30 | Turn upside down | 2058 | 391 | 83.1% | pick (18), move up (10) |
| 31 | Uncovering | 2426 | 312 | 72.8% | open (18), unfold (12) |
| 32 | Unfolding | 840 | 215 | 80.0% | fold (16), uncover (11) |

*Class 27 omitted (no data).*

---

## Appendix B — Reproducing the confusion matrix

```bash
cd /Data/romain.poggi/smth2smth
PYTHONPATH=src .venv/bin/python -m smth2smth.pipelines.evaluate \
  track=b training.checkpoint_path=checkpoints/track_b/vitl_fpc16ssv2_8f_lora_r16.pt
# For per-sample preds + CM, use the analysis script that produced
# logs/track_b_vjepa2_hfclf_8f_lora_r16_fullval_cm/
```

---

## References (internal)

- Run log: `logs/track_b_vjepa2_hfclf_8f_lora_r16_20260521_0339.log`
- Holdout FT log: `logs/track_b_vjepa2_hfclf_8f_lora_r16_val90_holdout_clf_20260522_val90clf.log`
- Experiment preset: `configs/experiment/track_b_vjepa2_hfclf_8f_lora_r16.yaml`
- Track B narrative: `report/track_b.tex` (Run 7), `track_b_next_steps.md` (§A–B directionality & 4-frame regime)
