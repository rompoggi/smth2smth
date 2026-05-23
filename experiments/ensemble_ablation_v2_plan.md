# Track A ensemble v2 — plan (correct TTA + combiner ablation)

**Status:** draft — **do not run until validated**  
**Supersedes for new runs:** holdout/submit rows that used `TTA_champion` (3-scale + flip, 6 views, **not** VideoMAE official).  
**Checkpoints:** `checkpoints/track_a/videomaev2+ft/s{42,43,44}_ft_val90_holdout.pt` (unchanged unless you re-upload).

---

## 1. What went wrong in v1 (P3 / P4 / P5)

| Item | v1 (already run) | v2 (this plan) |
|------|------------------|----------------|
| **TTA** | `champion`: scales `[0.857,1,1.143]` × **flip** (6 views) | **`official_2x3`**: `num_segment=2`, `num_crop=3`, **no flip** (6 views) |
| **Literature match** | E_R1 training eval habit | MCG-NJU / VideoMAE SSv2 **test: 2×3** (`new_ideas_tracka.md` L146) |
| **SSv2 direction** | Flip + remap (18↔19) may hurt | No flip (same as Track B E2 lesson) |
| **Combiners** | WS (`Mix`) + `Equal` only | WS, **CWS**, **LSG**, **mean**, **softmax-mean**, **majority** |

Cached v1 files stay under `outputs/ensemble/videomaev2_3seed/logits_*_champion.npy` for comparison; v2 uses a **new suffix** `official_2x3`.

---

## 2. TTA definitions (two inference outputs per member)

### 2.1 `Basic` — single-view (no TTA)

- One center crop, one temporal sample → **1 view**
- Cache: `logits_s{seed}_none.npy` (**already exists**)

### 2.2 `TTA` — official VideoMAE 2×3 (no flip)

- `num_segment=2`, `num_crop=3`, `flip_tta=false` (same as `TtaMode.DENSE_2X3` in code)
- **6 views** per clip; matches `test_num_segment=2`, `test_num_crop=3` in `new_ideas_tracka.md` RQ4

**Per-member aggregation (implement both for combiner fairness):**

| Cache tag | Per-member formula | Used for |
|-----------|-------------------|----------|
| `official_2x3_logits` | **Mean of logits** over 6 views | WS, CWS, LSG, **mean** (logit average) |
| `official_2x3_probs` | **Mean of softmax(logits)** over 6 views | **softmax-mean** combiner; aligns with `submit.py` `_predict_dense_tta` |

Note: `softmax(mean_logits) ≠ mean(softmax(logits))`. We report both as in RQ2(e).

**Not in v2 (unless you add later):**

- `champion` (3-scale + flip) — v1 only  
- `2×3×flip` (12 views) — Track A default in `configs/track/a.yaml`; discouraged on SSv2

---

## 3. Your three `Mix` protocols (fit vs eval logits)

These are **orthogonal** to combiner type (WS / CWS / …). Each cell in §5 is `{protocol} × {combiner}`.

| Protocol ID | Your name | Fit θ on holdout | Score / submit on holdout & test | v1 analogue |
|-------------|-----------|------------------|----------------------------------|-------------|
| **T** | `Mix(TTA)` | `TTA` (`official_2x3_*`) | `TTA` | P3 (but WS only, wrong TTA) |
| **BT** | `Mix(Basic) + TTA` | `Basic` (`none`) | `TTA` | P4 |
| **B** | `Mix(Basic)` | `Basic` | `Basic` | P5 |

**Rules**

- Learned combiners (WS, CWS, LSG): minimize holdout CE on the **fit** tensors; report metrics on the **eval** tensors (for BT, eval uses TTA with θ fitted on Basic).
- Fixed combiners: no fitting; eval = fit branch by definition.

---

## 4. Combiner catalog (`new_ideas_tracka.md` L67–88)

\(M=3\) members (s42, s43, s44), \(C=33\), holdout \(N=676\).

| ID | Name | Learned? | Combination rule | Parameters | Overfit risk |
|----|------|----------|------------------|------------|--------------|
| **Mean** | Equal / logit average | No | \(L = \frac{1}{M}\sum_i L_i\) | 0 | — |
| **Softmax** | Softmax-then-average (prob mean) | No | \(p = \frac{1}{M}\sum_i \mathrm{softmax}(L_i)\), \(\hat y = \arg\max p\) | 0 | — |
| **Vote** | Majority vote | No | \(\hat y = \mathrm{mode}_i(\arg\max L_i)\) | 0 | — |
| **WS** | Weighted sum (scalar) | Yes | \(L = \sum_i \theta_i L_i\), \(\theta \in \Delta^{M-1}\), fit by CE | \(M{-}1\) | Low |
| **CWS** | Class-dependent WS | Yes | \(L_c = \sum_i \theta_{i,c}\, L_{i,c}\), \(\theta_{\cdot,c} \in \Delta^{M-1}\) per class \(c\) | \(M(C{-}1)\) | Medium |
| **LSG** | Linear stacked generalization | Yes | \(\mathrm{vec}(L) = \sum_i W_i L_i\) or multinomial logistic on stacked logits | \(\mathcal{O}(MC^2)\) | **High** — use **L2** (sklearn `C` small) |

**WS** = your original `Mix` with simplex constraints (current `optimize_mix_weights`).  
**CWS / LSG** = Sen & Erdogan variants; doc recommends scalar WS when \(N\) is small — we still run CWS/LSG as ablations with regularization.

**Not running in v2 (optional later):** Caruana greedy integer selection (`pyensemble`), unconstrained WS without simplex (old A2/A3).

---

## 5. Full holdout experiment matrix

**Primary grid:** 3 protocols × 6 combiners = **18 rows** (+ singles).

### 5.1 Singles (sanity, no ensemble)

| Row | Eval TTA | Purpose |
|-----|----------|---------|
| S0-{42,43,44}-B | Basic | Per-seed baseline |
| S1-{42,43,44}-T | official_2x3 | Per-seed with correct TTA |

### 5.2 Ensemble rows (naming: `{Protocol}-{Combiner}`)

| Row | Protocol | Combiner | Fit | Eval |
|-----|----------|----------|-----|------|
| T-Mean | T | Mean | TTA logits | TTA logits |
| T-Softmax | T | Softmax | TTA probs* | TTA probs* |
| T-Vote | T | Vote | TTA logits | TTA logits |
| T-WS | T | WS | TTA logits | TTA logits |
| T-CWS | T | CWS | TTA logits | TTA logits |
| T-LSG | T | LSG | TTA logits | TTA logits |
| BT-Mean | BT | Mean | Basic | TTA |
| BT-Softmax | BT | Softmax | Basic | TTA probs* |
| BT-Vote | BT | Vote | Basic | TTA logits |
| BT-WS | BT | WS | Basic | TTA logits |
| BT-CWS | BT | CWS | Basic | TTA logits |
| BT-LSG | BT | LSG | Basic | TTA logits |
| B-Mean | B | Mean | Basic | Basic |
| B-Softmax | B | Softmax | Basic | Basic probs* |
| B-Vote | B | Vote | Basic | Basic |
| B-WS | B | WS | Basic | Basic |
| B-CWS | B | CWS | Basic | Basic |
| B-LSG | B | LSG | Basic | Basic |

\*For Softmax rows, members contribute **probability** vectors (TTA = `official_2x3_probs`); combiner averages probs. WS/CWS/LSG stay on **logits**.

### 5.3 Mapping to your three asks

| Your label | Rows to compare |
|------------|-----------------|
| `Mix(TTA)` | **T-WS** (also T-CWS, T-LSG, vs T-Mean) |
| `Mix(Basic)+TTA` | **BT-WS** (vs BT-Mean, T-WS) |
| `Mix(Basic)` | **B-WS** (vs B-Mean, T-WS) |

**Leaderboard pick:** best holdout Top-1 among `{T,BT,B} × {WS,CWS,LSG,Mean}` (and optionally Softmax if it wins Mean by >0.3 pt).

---

## 6. Cache layout (reuse + new)

Directory: `outputs/ensemble/videomaev2_3seed_v2/` (keep v1 untouched)

| File | Status |
|------|--------|
| `holdout_manifest.json`, `labels_holdout.npy` | Copy or symlink from v1 (same split_seed=42) |
| `logits_s{42,43,44}_none.npy` | **Reuse** from v1 (or copy) |
| `logits_s{42,43,44}_official_2x3_logits.npy` | **NEW** (~3 × forward holdout) |
| `logits_s{42,43,44}_official_2x3_probs.npy` | **NEW** (same passes, save probs) |
| `logits_test_s{42,43,44}_official_2x3_logits.npy` | **NEW** (~3 × 6913 test clips) |
| `logits_test_s{42,43,44}_official_2x3_probs.npy` | **NEW** (optional if submit uses logits only) |

**Runtime rough estimate (GPU):** ~2× v1 cache (6 holdout + 6 test passes vs 6 champion); ~30–45 min holdout + ~1–1.5 h test per seed depending on GPU.

---

## 7. Implementation checklist (before run)

- [ ] Rename / add `TtaMode.OFFICIAL_2X3` alias; dense path uses **VideoMAE pos_embed** fix (already in `inference.py` for champion scales; verify 224-only 2×3 needs no resize).
- [ ] Cache **both** logits and probs in one pass (save compute).
- [ ] `optimize.py`: `combine_probs`, `majority_vote`, `optimize_cws`, `optimize_lsg` (sklearn `LogisticRegression` multi_class on features `concat(L_1..L_M)` or block stack).
- [ ] `scripts/ensemble_track_a_videomae.py`: `optimize --grid v2`, read v2 cache dir; write `ensemble_results_v2.json`.
- [ ] `submit`: best row from v2, `official_2x3` test tensors, new CSV e.g. `submissions/track_a_ensemble_v2_best.csv`.
- [ ] Update `ensemble_ablation_tracka.md` §11 with v2 table (after run).

**LSG default (proposal):** `sklearn.linear_model.LogisticRegression(penalty='l2', C=1.0, max_iter=2000)` on features of shape `(N, M*C)` = flattened member logits; refit on holdout only for BT/B protocols as above.

**CWS default (proposal):** per-class \(\theta_{\cdot,c}\) on simplex, optimize sum of CE with softmax applied per-sample (same as WS but \(M\times C\) params with simplex constraint per class).

---

## 8. Execution commands (after you validate)

```bash
cd /Data/romain.poggi/smth2smth
VAL=/Data/thomas.turkieh/smth2smth/data/val   # adjust
TRAIN=/Data/thomas.turkieh/smth2smth/data/train
TEST=/Data/thomas.turkieh/smth2smth/data/test
CACHE=outputs/ensemble/videomaev2_3seed_v2

# Phase A — cache (holdout + test, official 2×3)
PYTHONPATH=src uv run python scripts/ensemble_track_a_videomae.py cache \
  --cache-dir "$CACHE" \
  --checkpoints-dir checkpoints/track_a/videomaev2+ft \
  --val-dir "$VAL" --train-dir "$TRAIN" \
  --tta official_2x3 --save-probs

# Phase B — holdout grid (instant once cached)
PYTHONPATH=src uv run python scripts/ensemble_track_a_videomae.py optimize \
  --cache-dir "$CACHE" --grid v2

# Phase C — submit best row only
PYTHONPATH=src uv run python scripts/ensemble_track_a_videomae.py submit \
  --cache-dir "$CACHE" --exp T-WS \
  --test-dir "$TEST" --train-dir "$TRAIN" \
  --output submissions/track_a_ensemble_v2_T-WS.csv
```

(Exact CLI flags to match implementation when you approve.)

---

## 9. Metrics & decision rules

**Primary:** holdout Top-1 (honest 676 clips, same split as v1).  
**Secondary:** holdout Top-5, CE, θ sparsity (is WS ≈ one-hot on s43?).

| Comparison | Question |
|------------|----------|
| T-WS vs T-Mean | Does learned WS beat equal weights with **correct** TTA? |
| T-WS vs BT-WS vs B-WS | Value of TTA and of θ/TTA mismatch |
| T-CWS vs T-WS | Worth extra parameters on 676 samples? |
| T-LSG vs T-WS | Overfit upper bound |
| T-Softmax vs T-Mean | RQ2(e) ~0.1–0.3 pt expectation |
| S1-* vs v1 P3 | Champion vs official TTA on same checkpoints |

**Submit:** one public CSV from best **T-*** row if any T-* beats best BT-* and B-*; else best overall. Expect **+0.3–1.0 pt** holdout vs v1 champion-TTA if flip was hurting.

---

## 10. Checkpoint caveat (unchanged)

- **s42:** `e_r1_ft_val90_holdout` (10-ep, honest holdout) — OK.  
- **s43 / s44:** non–E_R1 `resume_from` paths — ensemble still mostly **s43**. Fixing uploads is independent of this TTA/combiner grid.

---

## 11. Validation checklist for you

Please confirm:

1. **TTA = official 2×3, no flip** (not 2×3×flip, not champion scales)?  
2. **Full combiner grid** (18 rows) or a **minimal** subset (e.g. T/BT/B × {Mean, WS, CWS} only = 9 rows)?  
3. **New cache dir** `videomaev2_3seed_v2` OK?  
4. **Proceed with current s43/s44 checkpoints** or wait for re-uploads?

Reply e.g. “validate all / minimal grid / fix checkpoints first” and we start Phase A.
