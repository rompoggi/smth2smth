# Track A — learned-weight logit ensemble ablation (VideoMAE ViT-B, 3 seeds)

**Status:** code ready — run after all three checkpoints are uploaded  
**Scope:** ablation of stacking / learned-weight mixing from `experiments/new_ideas_tracka.md` (RQ2), on our best homogeneous ensemble: three fine-tunes that differ only by RNG seed.  
**Not in scope (this pass):** architecture-diverse members, train+val refit with frozen θ, Caruana greedy selection, per-class weights (CWS).

---

## 1. Base models (to upload)

| ID | Seed | Training recipe (expected) | Checkpoint path (fill when uploaded) | EMA preferred |
|----|------|----------------------------|--------------------------------------|---------------|
| **s42** | 42 | 200-ep SSL ViT-B → 60-ep FT (`champion_videomae`) → 10-ep holdout continuation on 90% official val | `checkpoints/track_a/videomaev2+ft/s42_ft_val90_holdout.pt` | EMA if present else `model_state_dict` |
| **s43** | 43 | same | `checkpoints/track_a/videomaev2+ft/s43_ft_val90_holdout.pt` | same |
| **s44** | 44 | same | `checkpoints/track_a/videomaev2+ft/s44_ft_val90_holdout.pt` | same |

**Holdout continuation** should match `track_a_ssl_finetune_e_r1_val90_holdout.yaml`:

- `dataset.use_official_val: true`
- `dataset.official_val_holdout_ratio: 0.1` (stratified per class; same split definition for all seeds — see §3)
- `training.epochs: 10`, low LR, resume from each seed’s 60-ep FT best

**Encoder:** shared 200-epoch SSL weights (one `*_encoder.pt` or three identical copies).  
**Head:** `model.head: attn` (E_R1 champion stack).  
**Frames:** 4, `tube_t: 1`, ViT-B.

When uploading, also record per checkpoint (from `extra` in `.pt` or log):

- `val_top1` on **honest 10% holdout** (the split used to fit θ)
- `val_top1` on full official val (informational only — do **not** use for θ)
- epoch index, `trained_class_indices` if present

---

## 2. Definitions

### 2.1 Logits and labels

For each video clip \(x\) in a set \(\mathcal{D}\):

- \(L_i(x) \in \mathbb{R}^{C}\): logits from member \(i\) (EMA weights, `eval()` mode).
- \(y(x) \in \{0,\ldots,C-1\}\): ground-truth class index on **holdout val only**.

\(C =\) `num_classes` (33 for Track A).

### 2.2 `Mix` — learned scalar weights (primary meta-learner)

Following RQ2 in `new_ideas_tracka.md` (stacked generalization with one weight per model):

\[
L_{\mathrm{mix}}(x) = \sum_{i=1}^{N} \theta_i \, L_i(x), \qquad \theta \in \Delta^{N-1}
\]

Constraints (default):

- \(\theta_i \ge 0\)
- \(\sum_i \theta_i = 1\) (probability simplex)

Optimization:

- Minimize **multiclass cross-entropy** between \(y\) and \(\mathrm{softmax}(L_{\mathrm{mix}})\) on the holdout val set.
- Solver: `scipy.optimize.minimize(..., method="SLSQP")` with simplex constraints (same as Kaggle “ensemble weight optimization” pattern).
- Init: \(\theta_i = 1/N\).

**Implementation note:** optimize in log-space or use softmax reparam \(\theta = \mathrm{softmax}(z)\) if SLSQP hits boundaries; report final \(\theta\) and holdout CE / Top-1 / Top-5.

**sklearn cross-check (optional row):** `sklearn.linear_model.LogisticRegression` on stacked logits `np.stack([L_1,\ldots,L_N], axis=1)` with `multi_class='multinomial'`, `fit_intercept=False`, and non-negative coefficients projected onto the simplex — should be close to SLSQP if regularization is off.

### 2.3 `Equal` — baseline combiner

\[
\theta_i = 1/N \quad \forall i
\]

No fitting. Always run as a baseline.

### 2.4 TTA modes (VideoMAE-safe)

**Critical:** default Track A `tta_scales: [0.875, 1.0, 1.125]` **crashes** VideoMAE ViT (`patch_size=16` needs divisible H,W). Use patch-safe scales from E_R1:

| Mode | Config | Views (approx.) | Use for |
|------|--------|-----------------|--------|
| **TTA_none** | `test.num_segment=1`, `num_crop=1`, `flip_tta=false`, no `tta_scales` | 1 | θ fitting for “plain” Mix; cheap val cache |
| **TTA_champion** | `training.tta=true`, `tta_flip=true`, `tta_scales=[0.857, 1.0, 1.143]` (from checkpoint cfg) | multi-scale × flip | Matches training-time val eval in `track_a_ssl_finetune_e_r1.yaml` |
| **TTA_2x3** | `test.num_segment=2`, `num_crop=3`, `flip_tta=false` | 6 | VideoMAE official SSv2-style (no flip on SSv2) |

Notation:

- **`M + TTA`**: run inference on \(M\) with the chosen TTA mode, **average softmax across views**, then take \(\log p\) (or keep probabilities and use CE on probs — be consistent everywhere).

**Order of operations (must be explicit in each experiment):**

1. **TTA → Mix:** compute TTA-averaged logits per member, then \(L_{\mathrm{mix}} = \sum_i \theta_i L_i^{\mathrm{TTA}}\).  
   Fit \(\theta\) on holdout with the **same** TTA mode used at test.

2. **Mix (plain) → TTA at test only:** fit \(\theta\) on **TTA_none** holdout logits; at submit, compute \(L_{\mathrm{mix}}^{\mathrm{test}} = \sum_i \theta_i L_i^{\mathrm{TTA}}\) (θ transferred from non-TTA val to TTA test). **Distribution shift** — include as ablation, not as primary “best practice”.

3. **Mix (plain) only:** fit and infer with **TTA_none** throughout.

There is no valid **`TTA(Mix(...))`** for separate checkpoints (cannot TTA a single fused model).

---

## 3. Validation protocol (θ must not leak)

### 3.1 Single holdout split for all seeds

Use one stratified 10% holdout of `data/val` (official val), fixed by seed:

```text
official_val_holdout_ratio: 0.1
split seed: 42   # same as train.py / split_train_val_stratified
```

All three members must be evaluated on **identical** holdout clip IDs. Export:

```text
outputs/ensemble/videomae_vitb_3seed/
  holdout_ids.json          # list of val sample keys
  labels_holdout.npy        # (N_holdout,)
  logits_s42_none.npy       # (N_holdout, C)
  logits_s42_champion.npy
  logits_s43_none.npy
  ...
```

**Forbidden for θ:** full val merged into training, test set, public LB feedback.

### 3.2 What the “10 epochs on holdout” models imply

Those checkpoints were trained on **90% of val**; the **10% holdout** was held out during that continuation. θ must be fit only on that 10% (honest). Reporting:

- **Primary metric:** holdout Top-1 (where Mix was fit).
- **Secondary:** full official val Top-1 (no θ fitting) — useful for sanity, not for selecting θ.

---

## 4. Experiment matrix

### 4.1 Primary runs (your requested comparisons)

| Exp ID | Members | θ fitting | Val logits | Test / submit logits | Notes |
|--------|---------|-----------|------------|-------------------|-------|
| **P0a** | s42 | — | TTA_none | TTA_none | single baseline |
| **P0b** | s42 | — | TTA_champion | TTA_champion | single + TTA |
| **P0c–d** | s43, s44 | — | same as P0a/b | same | repeat per seed |
| **P1** | s42,s43,s44 | Equal | TTA_none | TTA_none | seed-only diversity, no learning |
| **P2** | s42,s43,s44 | Equal | TTA_champion | TTA_champion | equal mix after per-model TTA |
| **P3** | s42,s43,s44 | **Mix** | TTA_champion per member | TTA_champion per member | **`Mix(s42+TTA, s43+TTA, s44+TTA)`** — fit θ on holdout TTA logits |
| **P4** | s42,s43,s44 | **Mix** | TTA_none | TTA_champion per member | **`Mix(s42,s43,s44)+TTA`** — θ from plain val, combine TTA test logits |
| **P5** | s42,s43,s44 | **Mix** | TTA_none | TTA_none | **`Mix(s42,s43,s44)`** — no TTA |

**Decision rule:** compare P3 vs P2 (does Mix beat Equal when TTA is aligned?), P5 vs P1 (Mix vs Equal without TTA), P4 vs P3 (cost of θ / TTA mismatch).

### 4.2 Ablation extensions (same checkpoints, low marginal cost)

| Exp ID | Variation | Purpose |
|--------|-----------|---------|
| **A1** | Mix + **TTA_2x3** (fit & test) | Official VideoMAE eval protocol; no flip |
| **A2** | Mix, **non-negative only** (no sum constraint) then renormalize | See if a member is down-weighted to ~0 |
| **A3** | Mix, **unconstrained** θ + renorm | Upper bound (risk meta-overfit on small holdout) |
| **A4** | **Softmax-then-average** vs logit Mix (P3) | Literature expects ~0.1–0.3 pt difference |
| **A5** | **Best single** vs **best θ-weighted** | If θ ≈ one-hot, ensemble not worth it |
| **A6** | sklearn `LogisticRegression` on stacked logits vs SLSQP | Implementation sanity check |
| **A7** | θ fit on **train+val 90%** OOF (optional, later) | Proper stacking — needs cached OOF logits; skip until Phase 2 |

### 4.3 Expected outcome bands (from literature)

On **seed-only** ensembles (highly correlated members), expect:

- Equal vs best single: **+0.5 to +1.5 pt** holdout
- Mix vs Equal: **+0.1 to +0.5 pt** (often collapses to near-equal θ)
- TTA_champion vs TTA_none: **+0.5 to +1.5 pt** per model
- P4 (θ mismatch) often **≤ P3** or slightly worse

If Mix θ is within **±0.02** of \(1/3\) for all seeds, report “θ ≈ equal” and stop tuning meta-learner.

---

## 5. Metrics and logging

For every row in §4, record on **holdout val**:

| Metric | Description |
|--------|-------------|
| Top-1 / Top-5 | Primary |
| CE | Same loss used to fit Mix |
| θ | `[θ_42, θ_43, θ_44]` |
| Δ vs P1 | Mix / Equal − 1/3 ensemble |
| Δ vs best P0 | Ensemble − max(s42,s43,s44) |
| Per-class accuracy | Optional JSON; spot-check direction classes |

For **Kaggle submit** (test):

- One CSV per exp ID: `submissions/track_a_ensemble_<ExpID>.csv`
- Log `test_cfg`, checkpoint paths, θ vector in `outputs/ensemble/.../manifest.yaml`

---

## 6. Execution pipeline (implementation checklist)

Execute in order once checkpoints exist.

### Phase 0 — Inventory (you)

- [ ] Upload three final `.pt` (EMA state preferred) + confirm shared SSL encoder path
- [ ] Confirm all three used **identical** `official_val_holdout_ratio` and split seed
- [ ] Fill checkpoint table in §1

### Phase 1 — Cache holdout logits

```bash
cd /Data/romain.poggi/smth2smth
PYTHONPATH=src uv run python scripts/ensemble_track_a_videomae.py cache \
  --checkpoints-dir checkpoints/track_a/videomaev2+ft \
  --cache-dir outputs/ensemble/videomaev2_3seed
```

Writes `logits_s{42,43,44}_{none,champion}.npy`, `labels_holdout.npy`, `holdout_manifest.json`.

**Requires all three** `sXX_ft_val90_holdout.pt` files. Re-run with `--force` to refresh.

**Runtime:** ~6 forward passes (3 seeds × 2 TTA modes); champion TTA is ~6 views per clip.

### Phase 2 — Offline Mix / Equal (P0–P5)

```bash
PYTHONPATH=src uv run python scripts/ensemble_track_a_videomae.py optimize \
  --cache-dir outputs/ensemble/videomaev2_3seed
# or one row:  ... optimize --exp P3
```

Output: `outputs/ensemble/videomaev2_3seed/ensemble_results.json`

### Phase 3 — Submit

**Not implemented yet** (`submit` subcommand exits with a message). For a single member, use `smth2smth.pipelines.submit` with one checkpoint. Combined test CSV will need a follow-up script.

**Submit priority order (when implemented):**

1. P2 (Equal + TTA_champion)  
2. P3 (Mix + TTA aligned)  
3. P5, P4  

---

## 7. TTA / submit configuration templates

### 7.1 TTA_none (val cache + P5)

```yaml
test:
  num_segment: 1
  num_crop: 1
  flip_tta: false
# Do not set training.tta_scales on submit
```

### 7.2 TTA_champion (P2, P3 — align with E_R1 eval)

Use each checkpoint’s saved `cfg` (from `cfg_from_checkpoint`):

```yaml
training:
  tta: true
  tta_flip: true
  tta_scales: [0.857, 1.0, 1.143]
```

Ensure `submit.py` path uses the same multi-scale + flip logic as training eval (not only `test.num_segment`).

### 7.3 TTA_2x3 (A1)

```yaml
test:
  num_segment: 2
  num_crop: 3
  flip_tta: false
```

---

## 8. Report integration

Add a short subsection to Track A report (ensemble ablation):

- Table: P0–P5 holdout Top-1, θ, Δ vs Equal  
- One paragraph: seed-only diversity limits (cite Caruana / Model Soups expectation from `new_ideas_tracka.md`)  
- Note TTA scale patch-divisibility constraint for VideoMAE  
- If θ ≈ equal, state that **Mix did not beat Equal** on honest holdout

---

## 9. Open questions (resolve when checkpoints arrive)

1. **Exact checkpoint filenames** and whether final weights are **EMA** or live.  
2. **Shared holdout split:** confirm `seed=42` for `split_train_val_stratified` in all three training configs.  
3. **SSL encoder:** one `e_r1_encoder.pt` or per-seed? (Should not affect logits if FT converged.)  
4. **Submit TTA:** confirm training eval uses `evaluate_epoch` + `training.tta` (not `test.*`) so P3 matches how val was logged during the 10-ep holdout run.  
5. **Public LB:** select **one** primary submit for the competition; keep others as ablation only.

---

## 10. Quick reference — your three formulas

| Name | Formal expression | Exp ID |
|------|-------------------|--------|
| `Mix(s42+TTA, s43+TTA, s44+TTA)` | Fit \(\theta\) on holdout with \(L_i = L_i^{\mathrm{TTA\_champion}}\); test \(\sum_i \theta_i L_i^{\mathrm{TTA\_champion}}\) | **P3** |
| `Mix(s42,s43,s44)+TTA` | Fit \(\theta\) on holdout with \(L_i = L_i^{\mathrm{none}}\); test \(\sum_i \theta_i L_i^{\mathrm{TTA\_champion}}\) | **P4** |
| `Mix(s42,s43,s44)` | Fit and test with \(L_i^{\mathrm{none}}\) only | **P5** |

**Equal** counterparts: **P2** (TTA_champion), **P1** (none).

---

## 11. Results (2026-05-22, N=676 holdout)

Cached logits: `outputs/ensemble/videomaev2_3seed/logits_s{42,43,44}_{none,champion}.npy`

| Exp ID | holdout Top-1 | holdout Top-5 | θ (s42, s43, s44) | Notes |
|--------|---------------|---------------|-------------------|-------|
| P5 | 59.91 | 86.83 | ≈0, 0.693, 0.307 | Mix on plain val |
| P4 | 60.50 | 88.17 | ≈0, 0.693, 0.307 | θ fit plain; **eval** on champion TTA |
| P3 | 61.24 | 88.02 | ≈0, 0.796, 0.204 | Mix on champion TTA (aligned) |

θ heavily favors **s43**; s42 ≈ 0. Singles (from cache): s42 none 53.6%, champion ~55%; see checkpoint `extra.val_top1`.

---

*Derived from `experiments/new_ideas_tracka.md` §RQ2 and champion configs `champion_videomae`, `track_a_ssl_finetune_e_r1`, `track_a_ssl_finetune_e_r1_val90_holdout`.*
