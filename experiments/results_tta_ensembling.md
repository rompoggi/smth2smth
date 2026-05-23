# Track A — TTA sweeps & logit ensembling: consolidated results

**Last updated:** 2026-05-23  
**Scope:** VideoMAE ViT-B Track A — test-time augmentation (TTA), holdout evaluation, 3-seed logit ensembles, and Kaggle public leaderboard (LB) submissions.  
**Primary checkpoints (ensemble):** `checkpoints/track_a/videomaev2+ft/s{42,43,44}_ft_val90_holdout.pt`

This document merges experiment plans, logs, cached logits, holdout metrics, confusion-matrix analysis, and the Kaggle submission history (screenshot 2026-05-23). Numbers are cited to **file paths** in this repo unless noted as user-reported.

---

## Executive summary

| Layer | What worked | What did not |
|-------|-------------|--------------|
| **TTA (single model)** | **3-scale + horizontal flip** (`scales3_flip` / `TTA_champion`: `[0.857,1,1.143]` + flip, 6 views) — best on s43 holdout sweep (~54.1% top-1, split seed **43**) and strong Kaggle singles (~0.465–0.466). | **Official VideoMAE 2×3, no flip** — catastrophic on holdout (~46% per seed) and LB **0.3277** on E_R1 submit. **Multi-scale without flip** ≈ no gain vs plain in sweep. |
| **Ensembling (3 seeds)** | **Mix + aligned champion TTA (P3 / T-ws):** holdout **61.24%**, LB **0.4717** (best submission). θ ≈ **(0, 0.80, 0.20)** — effectively **s43 + a little s44**. | **Mix on official_2x3 TTA (v2 T-*):** holdout ~46% (worse than no TTA). **LSG combiner:** inflated holdout (75%+) — overfit, not for submit. **θ mismatch (P4):** holdout 60.50% < P3. |

**Best public LB (repo evidence):** `track_a_ensemble_P3.csv` → **0.4717** (+0.57 pt vs cited prior single ~0.4660).

---

## Part A — Repository index (where TTA / ensembling is documented)

Use this table to find primary sources. Paths are relative to repo root `/Data/romain.poggi/smth2smth`.

### A.1 Experiment plans & results write-ups

| Path | Content |
|------|---------|
| `experiments/ensemble_ablation_tracka.md` | **v1 ensemble** (P1–P5), TTA modes, holdout protocol, **§11 results** (P3/P4/P5), cache/submit commands |
| `experiments/ensemble_ablation_v2_plan.md` | **v2 grid** (T/BT/B × 6 combiners), official `2×3` TTA definition, v1 vs v2 comparison |
| `experiments/new_ideas_tracka.md` | RQ2 stacking / Mix, combiner catalog (WS, CWS, LSG), official SSv2 FT recipe (`test_num_segment=2`, `num_crop=3`) |
| `experiments/instruction_hc_ablation.md` | HC ablation: submit TTA `tta_scales=[0.857,1,1.143]` + flip |
| `experiments/experiments.md` | Historical Track A notes (e.g. rouget 36% LB with `tta_scales=[1.0]`) |
| `experiments/experiment_16052026.md` | Submit-time hygiene |
| **`experiments/results_tta_ensembling.md`** | **This file** |

### A.2 LaTeX report

| Path | Content |
|------|---------|
| `report/track_a.tex` | Phase 2 TTA (§`\ref{sec:phase2:tta}`), Phase 4 multi-scale TTA, E_R1 / champion FT narrative, Kaggle notes |
| `report/track_a_inventory.tex` | Submission inventory: wrong `tta_scales=[1.0]` vs patch-safe scales, flip remap |
| `report/track_b.tex` | Track B TTA (2×3 no-flip lesson; contrast with Track A) |

### A.3 Code (implementation)

| Path | Content |
|------|---------|
| `src/smth2smth/ensemble/inference.py` | `TtaMode`: `none`, `champion`, `official_2x3`; champion defaults `[0.857,1,1.143]` + flip |
| `src/smth2smth/ensemble/optimize.py` | Scalar **WS** (`optimize_mix_weights`, SLSQP simplex) |
| `src/smth2smth/ensemble/combiners.py` | v2 combiners: mean, softmax, vote, WS, CWS, LSG |
| `src/smth2smth/ensemble/holdout.py` | Stratified 10% val holdout (`split_seed=42` default) |
| `src/smth2smth/ensemble/submit.py` | Test CSV from cached logits + combiner weights |
| `src/smth2smth/pipelines/submit.py` | Dense 2×3 TTA, multi-scale ViT, flip-pair remap (18↔19) |
| `scripts/ensemble_track_a_videomae.py` | `cache` / `optimize` / `submit` CLI |
| `scripts/plot_holdout_confusion_matrix.py` | Holdout confusion matrix for grid rows |
| `tests/pipelines/test_submit_tta.py` | Submit TTA unit tests |

### A.4 Configs

| Path | TTA-relevant settings |
|------|---------------------|
| `configs/track/a.yaml` | Default Track A: `tta_scales: [0.875, 1.0, 1.125]` (**unsafe for ViT patch 16** — see inventory) |
| `configs/experiment/track_a_ssl_finetune_e_r1.yaml` | **Champion:** `tta_scales: [0.857, 1.0, 1.143]`, `tta_flip: true` |
| `configs/experiment/track_a_ssl_finetune_e_r1_val90_holdout.yaml` | 10-ep val90 continuation; inherits E_R1 TTA from parent |
| `configs/experiment/track_a_hc_ablation_*.yaml` | Same champion scales + flip at submit |
| `configs/test/vjepa2_official_2x3.yaml` | Track B style 2×3 reference |

### A.5 Cached artifacts & metrics

| Path | Content |
|------|---------|
| `outputs/ensemble/videomaev2_3seed/` | **v1:** `logits_s*_none.npy`, `logits_s*_champion.npy`, `ensemble_results.json` |
| `outputs/ensemble/videomaev2_3seed_v2/` | **v2:** `logits_s*_official_2x3_*.npy`, `ensemble_results_v2.json` |
| `outputs/ensemble/videomaev2_3seed_v3_champion/` | **v3:** copy of v1 champion caches + full combiner grid `ensemble_results.json` |
| `outputs/ensemble/analysis/T_ws/` | Confusion matrix + `summary.json` for T-ws |

### A.6 Logs (TTA / ensemble runs)

| Path | Content |
|------|---------|
| `logs/ensemble_cache_videomaev2_3seed.log` | v1 cache: champion + none per seed; flip remap lines |
| `logs/ensemble_optimize_P5_P4_P3.log` | v1 P3/P4/P5 optimize |
| `logs/ensemble_submit_P3.log` | P3 test champion logits + CSV |
| `logs/ensemble_v2_cache_20260519.log` | v2 official_2x3 GPU cache (676 holdout, 6913 test) |
| `logs/ensemble_v2_optimize_20260519.log` | v2 18-row grid (partial run + sklearn LSG fix) |
| `logs/ensemble_v2_submit_20260519.log` | v2 B-ws submit |
| `logs/ensemble_v3_cache_champion.log` | v3: copied v1 caches (no GPU) |
| `logs/ensemble_v3_optimize_champion.log` | v3 full grid, T-ws best |
| `logs/ensemble_v3_submit_T-ws.log` | v3 T-ws CSV |
| `logs/sole_e_r1_ft_val90_holdout_20260518.log` | E_R1 10-ep holdout training |
| `logs/sole_e_r1_submit_20260518.log` | E_R1 submit run |
| `logs/hc/thon_hc_baseline_s43_20260517.log` | HC baseline s43 training |
| `logs/hc/truite_hc_shc_s44_20260518.log` | HC SHC s44 training |

### A.7 Submissions (local CSV; Kaggle names from screenshot)

| Path | Notes |
|------|--------|
| `submissions/track_a_ensemble_P3.csv` | **Best LB 0.4717** (screenshot) |
| `submissions/track_a_ensemble_v3_T-ws.csv` | Same recipe as P3; may not be uploaded under this name |
| `submissions/track_a_ensemble_v2_B-ws.csv` | v2 best (no TTA); holdout-only selection |
| Other `track_a_hc_*`, `track_a_e_r1_*` | See §5 — many on LB screenshot but not all present under `submissions/` in repo |

### A.8 External / missing in repo

| Item | Status |
|------|--------|
| `logs/hc/tta_holdout_sweep_results.json` | Referenced in agent run; **not found** on disk at doc time — sweep results below are from **session log** (user query 2026-05-23) |
| Kaggle LB scores | **Screenshot** (2026-05-23 02:03) — not all scores duplicated in committed logs |

---

## Part B — TTA definitions (code vs literature)

### B.1 Modes implemented in `ensemble/inference.py`

| Mode | Config / behavior | Views (typical) | Source |
|------|-------------------|-----------------|--------|
| **`none`** | Single center crop, one temporal sample | 1 | `TtaMode.NONE` |
| **`champion`** | `tta_scales` from checkpoint or default **`[0.857, 1.0, 1.143]`**, `tta_flip=true`, flip-pair remap | 3 scales × 2 flip states → **6** (averaged logits) | `inference.py` L43–51; `track_a_ssl_finetune_e_r1.yaml` |
| **`official_2x3`** | `num_segment=2`, `num_crop=3`, **no flip** | **6** | `inference.py` L49–50; VideoMAE SSv2 test recipe in `new_ideas_tracka.md` L146 |

**Per-view aggregation (current code):** mean of logits over views before caching (champion); official_2x3 also stores mean logits + mean softmax (`probs_*`).

**Not cached in ensemble runs:** `2×3×flip` (12 views) — discouraged for SSv2 direction (`ensemble_ablation_v2_plan.md` §2.2).

### B.2 Naming map (filenames ↔ modes)

| Filename / log tag | Meaning |
|--------------------|---------|
| `tta3flip`, `scales3_flip` | Champion: 3 scales + flip |
| `notta`, `no_tta` | `TtaMode.NONE` at submit |
| `official_2x3_noflip` | Dense 2×3, no flip |
| `_champion` (cache suffix) | `TtaMode.CHAMPION` |

### B.3 Patch-safe scales (VideoMAE ViT-B, patch 16)

- **Use:** `[0.857, 1.0, 1.143]` → 192 / 224 / 256 px (`ensemble_ablation_tracka.md` §2.4).
- **Avoid:** `[0.875, 1.0, 1.125]` on ViT (`configs/track/a.yaml`) — crashes or wrong grids (`report/track_a_inventory.tex`).

---

## Part C — Holdout protocol (all ensemble & sweep comparisons)

| Parameter | Value | Source |
|-----------|-------|--------|
| Val root | `/Data/thomas.turkieh/smth2smth/data/val` (cluster) | `ensemble_track_a_videomae.py` |
| Holdout ratio | 10% stratified per class | `holdout.py`, `official_val_holdout_ratio: 0.1` |
| **Split seed** | **42** (ensemble, v1/v2/v3) | `ensemble_ablation_tracka.md` §3.1; `logs/ensemble_cache_*.log` |
| Holdout size | **N = 676** clips | All `ensemble_results*.json`, cache logs |
| θ fitting | Minimize holdout CE, simplex WS | `optimize.py` |

**Important:** The **s43 TTA sweep** (below) used `holdout_ratio=0.1`, **`split_seed=43`** — same ratio, **different clips** than ensemble. Do not compare 54.14% sweep vs 61.24% P3 directly without aligning splits.

---

## Part D — TTA holdout sweep (single checkpoint)

**Checkpoint:** `baseline_s43_ft_val90_holdout.pt` (HC baseline arm, seed 43, 10-ep val90 holdout style).  
**Protocol:** User-reported run ending with `logs/hc/tta_holdout_sweep_results.json` (file **not** in repo at doc time).  
**Holdout:** n=676, ratio=0.1, **seed=43**.

### D.1 Ranked results (holdout top-1)

| Rank | Config ID | top-1 | top-5 | flip | scales |
|------|-----------|-------|-------|------|--------|
| 1 | **scales3_flip** | **0.5414** | 0.6243 | true | [0.857, 1.0, 1.143] |
| 1 | **scales875_flip** | **0.5414** | 0.6243 | true | [0.875, 1.0, 1.125] |
| 3 | flip | 0.5325 | 0.6154 | true | [1.0] |
| 4 | none | 0.5222 | 0.6021 | false | [1.0] |
| 5 | scales3 | 0.5222 | 0.6050 | false | [0.857, 1.0, 1.143] |

**Source:** User/agent session log (2026-05-23); ViT logs showed 0.857 and 0.875 both map to **192×192**.

### D.2 Interpretation

1. **Flip alone:** +1.0 pt over none (52.22% → 53.25%).
2. **3-scale alone:** **0 pt** vs none — scale jitter useless without flip on this checkpoint/split.
3. **3-scale + flip:** +1.9 pt over none; +0.9 pt over flip-only — **production TTA = `scales3_flip`**.

This matches **`TtaMode.CHAMPION`** used in ensemble v1/v3 (`logs/ensemble_cache_videomaev2_3seed.log`: flip remap `[(18, 19)]`).

---

## Part E — Three-seed ensemble experiments

### E.1 Base checkpoints

| Seed | Path | Training note | Source |
|------|------|---------------|--------|
| 42 | `checkpoints/track_a/videomaev2+ft/s42_ft_val90_holdout.pt` | E_R1-style 10-ep on 90% val | `ensemble_ablation_tracka.md` §1 |
| 43 | `checkpoints/track_a/videomaev2+ft/s43_ft_val90_holdout.pt` | HC baseline FT; **best single** | Same; θ dominance |
| 44 | `checkpoints/track_a/videomaev2+ft/s44_ft_val90_holdout.pt` | HC SHC / related; non–E_R1 resume in notes | `ensemble_ablation_v2_plan.md` §10 |

All three: VideoMAE ViT-B, 4 frames, shared SSL encoder, **attn head** (`ensemble_ablation_tracka.md`).

### E.2 v1 — Protocols P1–P5 (`outputs/ensemble/videomaev2_3seed`)

**Cache:** `logs/ensemble_cache_videomaev2_3seed.log` — `logits_s{42,43,44}_{none,champion}.npy`, test `logits_test_s*_champion.npy`.

| Exp | Description | Holdout top-1 | Holdout top-5 | θ (s42, s43, s44) | Source |
|-----|-------------|---------------|---------------|-------------------|--------|
| **P5** | Mix(Basic) — fit & test `none` | **59.91%** | 86.83% | ≈0, 0.693, 0.307 | `ensemble_results.json` |
| **P4** | Mix(Basic)+TTA — fit `none`, test `champion` | **60.50%** | 88.17% | ≈0, 0.693, 0.307 | Same |
| **P3** | **Mix(TTA)** — fit & test `champion` | **61.24%** | 88.02% | ≈0, **0.796**, **0.204** | Same; `ensemble_ablation_tracka.md` §11 |

**Submit:** `submissions/track_a_ensemble_P3.csv` — `logs/ensemble_submit_P3.log`  
- Test TTA: champion per seed, flip remap  
- Weights: `[~0, 0.796, 0.204]`  
- **Kaggle public LB: 0.4717** (screenshot 2026-05-23) — best in submission list  
- Cited improvement vs ~**0.4660** single E_R1 no-TTA submit (+0.57 pt) from project notes

**What worked:** Aligned TTA + WS; down-weight s42; emphasize s43.  
**What did not:** P4 θ/TTA mismatch still below P3; equal-weight TTA (P2, not fully logged in JSON) expected ~58% band.

### E.3 v2 — Official 2×3 + 18-row grid (`outputs/ensemble/videomaev2_3seed_v2`)

**Cache:** `logs/ensemble_v2_cache_20260519.log` — GPU run, official_2x3 logits+probs, holdout N=676, test N=6913.

**TTA failure on holdout (single-seed T = official_2x3):**

| Row | Holdout top-1 | Source |
|-----|---------------|--------|
| S1-s43-T | **46.60%** | `logs/ensemble_v2_optimize_20260519.log` |
| S1-s42-T | 39.94% | Same |
| S1-s44-T | 43.64% | Same |
| S0-s43-B (no TTA) | **58.88%** | Same |

**Ensemble with official TTA (broken):**

| Row | Holdout top-1 |
|-----|---------------|
| T-ws | 45.86% |
| BT-ws | 46.15% |
| **B-ws** | **59.91%** (same as P5 — no TTA) |

**Source:** `ensemble_results_v2.json`, `logs/ensemble_v2_optimize_20260519.log`

**Submit:** `submissions/track_a_ensemble_v2_B-ws.csv` (Basic only; not top LB row).

**Conclusion:** **Official 2×3 without flip is the wrong TTA** for these checkpoints on our holdout, despite literature defaults (`new_ideas_tracka.md`, VideoMAE SSv2).

### E.4 v3 — Champion TTA + full combiner grid (`outputs/ensemble/videomaev2_3seed_v3_champion`)

**Cache:** Copied v1 champion/none/test caches — `logs/ensemble_v3_cache_champion.log` (no GPU re-forward).

**Best rows (holdout, excluding LSG overfit):**

| Row | Holdout top-1 | Notes | Source |
|-----|---------------|-------|--------|
| S1-s43-T | **61.39%** | Best **single** + champion | `ensemble_results.json` v3 |
| **T-ws** | **61.24%** | = P3 recipe; θ ≈ (0, 0.796, 0.204) | `logs/ensemble_v3_optimize_champion.log` |
| T-cws | 60.80% | Below T-ws | Same |
| BT-ws | 60.50% | = P4 | Same |
| B-ws | 59.91% | = P5 | Same |

**Submit:** `submissions/track_a_ensemble_v3_T-ws.csv` — `logs/ensemble_v3_submit_T-ws.log` (6913 rows; byte-identical holdout caches to v1 P3).

**LSG rows (75–76% holdout):** Severe meta-overfit on N=676 — **do not use for LB** (`combiners.py` ridge fix; still not calibrated for selection).

---

## Part F — Kaggle public leaderboard (submission screenshot)

**Source:** User screenshot `Screenshot_2026-05-23_at_02.03.44` (12 submissions, sorted by score).

| Rank | Submission file | Public LB top-1 | TTA (from name) | Checkpoint family |
|------|-----------------|-----------------|-----------------|-------------------|
| 1 | `track_a_ensemble_P3.csv` | **0.4717** | champion (3-seed Mix) | s42+s43+s44 `*_ft_val90_holdout` |
| 2 | `track_a_hc_baseline_s43_val90_holdout_20260517.csv` | 0.4657 | notta (name) | HC baseline s43 |
| 3 | `track_a_hc_baseline_s43_val90_holdout_tta3flip_20260517.csv` | 0.4654 | **tta3flip** | HC baseline s43 |
| 4 | `track_a_e_r1_ft_val90_holdout_no_tta.csv` | 0.4660 | no_tta | E_R1 10-ep holdout |
| 5 | `track_a_hc_baseline_s44_val90_holdout_best_notta_20260519.csv` | 0.4628 | notta | HC baseline s44 |
| 6 | `track_a_hc_shc_s44_val90_holdout_tta3flip_20260521.csv` | 0.4602 | tta3flip | HC SHC s44 |
| 7 | `track_a_hc_shc_s44_val90_holdout_20260519.csv` | 0.4518 | (default/champion in recipe) | HC SHC s44 |
| 8 | `track_a_hc_shc_s42_val90_holdout_20260519.csv` | 0.4512 | — | HC SHC s42 |
| 9–11 | `track_a_hc_baseline_s44_*_notta_*.csv` | 0.4506–0.4570 | notta | HC baseline s44 variants |
| 12 | `track_a_e_r1_ft_val90_holdout_official_2x3_noflip.csv` | **0.3277** | official 2×3 | E_R1 — **failed** |

### F.1 LB takeaways

1. **Ensemble P3 (champion TTA + WS)** is **+0.57 pt** above the cited E_R1 no-TTA single (0.4660) and above all listed HC singles.
2. **tta3flip vs notta** on the **same** checkpoint: tiny swing (e.g. s43: 0.4657 vs 0.4654; s44: 0.4602 vs 0.4518) — TTA helps some seeds on LB, not others.
3. **official_2x3_noflip** is a clear **outlier failure** on LB — consistent with v2 holdout collapse.

---

## Part G — Holdout error analysis (T-ws / P3)

**Artifact:** `outputs/ensemble/analysis/T_ws/`  
**Script:** `scripts/plot_holdout_confusion_matrix.py`  
**Model:** T-ws = same as P3 (champion TTA, θ ≈ s43-only)

| Metric | Value |
|--------|-------|
| Holdout top-1 | 61.24% |
| Macro recall (approx.) | ~56% |

**Strong classes (recall):** Moving away (89%), Moving down (87%), Spilling (83%).  
**Weak classes:** Picking (15%), Pull R (17%), Pretend-pour (17%), Put onto (36%).  
**Systematic confusion:** Pull L ↔ Pull R (~50% cross); pretend vs real pour/pick/put.

**Implication:** Global TTA+WS does not fix **direction** or **fine-grained verb** errors; see `summary.json` in analysis folder.

---

## Part H — Decision log (what to run going forward)

| Recommendation | Rationale |
|----------------|-----------|
| **Submit champion TTA** (`scales3_flip` / P3 / T-ws) | Best holdout + best public LB in repo |
| **Do not submit official_2x3** on these VideoMAE checkpoints | Holdout & LB evidence |
| **Keep holdout split seed=42** for θ | Consistency with all ensemble caches |
| **Ignore LSG combiner** for selection | Holdout inflation |
| **Optional:** view-level WS inside TTA (not run yet) | Theoretical +0.3–1 pt; needs per-view logit cache |
| **Optional:** no-flip ablation on submit for direction classes | Track B lesson; not yet A/B on LB |

---

## Part I — Command reference

```bash
# Champion cache (or copy from v1)
PYTHONPATH=src uv run python scripts/ensemble_track_a_videomae.py cache \
  --cache-dir outputs/ensemble/videomaev2_3seed_v3_champion \
  --tta-mode champion --tta-tag champion

# Full grid
PYTHONPATH=src uv run python scripts/ensemble_track_a_videomae.py optimize \
  --cache-dir outputs/ensemble/videomaev2_3seed_v3_champion

# Submit best row
PYTHONPATH=src uv run python scripts/ensemble_track_a_videomae.py submit \
  --cache-dir outputs/ensemble/videomaev2_3seed_v3_champion \
  --exp T-ws --output submissions/track_a_ensemble_v3_T-ws.csv

# Confusion matrix
PYTHONPATH=src uv run python scripts/plot_holdout_confusion_matrix.py --exp T-ws
```

---

## Part J — Glossary

| Symbol | Meaning |
|--------|---------|
| **WS / Mix** | Learned scalar weights on logit vectors, simplex constraint |
| **champion / scales3_flip** | 3 spatial scales + horizontal flip TTA (6 views) |
| **official_2x3** | 2 temporal segments × 3 crops, no flip |
| **θ** | Ensemble weights across seeds (s42, s43, s44) |
| **Holdout** | 10% stratified official val; N=676 for split_seed=42 |

---

## References (internal)

1. `experiments/ensemble_ablation_tracka.md` — v1 design & P3/P4/P5 results  
2. `experiments/ensemble_ablation_v2_plan.md` — v2 official_2x3 grid  
3. `experiments/new_ideas_tracka.md` — RQ2, official FT/TTA recipe  
4. `outputs/ensemble/videomaev2_3seed/ensemble_results.json` — P3 metrics  
5. `outputs/ensemble/videomaev2_3seed_v2/ensemble_results_v2.json` — v2 grid  
6. `outputs/ensemble/videomaev2_3seed_v3_champion/ensemble_results.json` — v3 grid  
7. `logs/ensemble_cache_videomaev2_3seed.log`, `logs/ensemble_v2_optimize_20260519.log`, `logs/ensemble_v3_optimize_champion.log`  
8. `logs/ensemble_submit_P3.log` — P3 submit  
9. `report/track_a.tex`, `report/track_a_inventory.tex` — TTA implementation & submission pitfalls  
10. User screenshot 2026-05-23 — Kaggle LB table  
11. User/agent session — s43 `tta_holdout_sweep` ranked table (§D)
