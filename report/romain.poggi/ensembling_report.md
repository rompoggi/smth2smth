# Ensembling MAE500 diverse heads — results

**Setup.** Fixed VideoMAEv2 ViT-B backbone (SSL ep500 on the provided clips),
stabilised fine-tune recipe, three classifier heads (mean-pool, Perceiver-Q,
Divided-Space-Time-K) over seeds 42–44, train-only protocol (clean official val,
N=6745). All combiner/diversity numbers below are **5-fold stratified
out-of-fold (OOF)**: the combiner is fit on 4 folds and scored on the held fold,
pooled over all 6745 clips, with 95 % bootstrap CIs. Test/LB numbers are the
disjoint Kaggle set (N=6913), reported for **both** basic (no-TTA) and
champion-TTA inference; the val axis is basic OOF throughout.

All figures live in `outputs/ensemble/mae500_stab_val/plots/`.

## 1. Combiner choice does not matter — lock softmax

We compare six combiners on a 4-member diverse set (mean-pool + Q8 + DivST-K9 +
Q16): simple average, **softmax**-average, majority vote, weighted-sum (WS),
class-weighted (CWS) and full linear stacking (LSG).

![combiner comparison](../../outputs/ensemble/mae500_stab_val/plots/combiner_bars.png)

| combiner | val top-1 | 95 % CI |
|----------|-----------|---------|
| LSG      | 56.66 | [55.40, 57.86] |
| softmax  | 55.95 | [54.57, 57.12] |
| CWS      | 55.94 | [54.64, 57.06] |
| mean     | 55.73 | [54.39, 56.91] |
| WS       | 55.67 | [54.34, 56.78] |
| vote     | 55.34 | [53.99, 56.57] |

The CIs (~±1.2 pp) all overlap: **no combiner is significantly better.** Even
under proper OOF the learned combiners do not pay off — LSG is nominally top but
not separable from the rest, and the fitted weights are not uniform yet add
nothing:

![learned coefficients](../../outputs/ensemble/mae500_stab_val/plots/learned_coeffs.png)

WS tilts toward Q8 (0.32) and starves the dominated DivST-K9 head (0.14); CWS
learns noisy per-class member preferences (e.g. mean-pool dominant on class 14,
DivST-K9 ≈ 0 there). Neither beats plain softmax on held-out data. **We lock
softmax** (parameter-free, cannot overfit) for everything that follows.

## 2. Seed diversity ≈ architecture diversity

With the combiner fixed, we size-match three member sets and measure the
ensemble gain over the best single member.

![diversity grid](../../outputs/ensemble/mae500_stab_val/plots/diversity_grid.png)

| member set | ensemble top-1 | gain vs best single |
|------------|----------------|---------------------|
| seed-only (Q8 × s42/s43/s44) | 55.63 | **+1.32 pp** |
| architecture-only (mean-pool / Q8 / DivST-K9, s42) | 55.64 | **+1.33 pp** |
| all-axes (4 heads × 3 seeds) | 55.24 | +1.54 pp |

Seed-only and architecture-only are **identical** (55.63 vs 55.64). Ensembling
buys ~+1.3 pp regardless of the diversity axis — varying the architecture does
**not** beat simply varying the seed of the single best head.

## 3. Why: decorrelation does not convert to accuracy

![disagreement heatmap](../../outputs/ensemble/mae500_stab_val/plots/disagreement_heatmap.png)

*Members ordered so the three Q8 seeds form the red-boxed block. The seed block
is slightly more correlated / less disagreeing than the cross-architecture pairs
— architectural mixing decorrelates marginally more, but the effect is small.*

| triple | mean pairwise disagreement | mean error-correlation |
|--------|----------------------------|------------------------|
| architecture (mp/Q8/DivST-K9) | 0.275 | 0.711 |
| seed (Q8 × 3) | 0.262 | 0.729 |

Mixing architectures *is* marginally more decorrelated (higher disagreement,
lower error-correlation), but it yields **no extra ensemble accuracy** — the
extra disagreement comes from the weaker, accuracy-dominated heads (mean-pool,
DivST) being wrong in different ways, which does not add correct votes.

## 4. Gain vs backbone strength

![gain vs SSL epoch](../../outputs/ensemble/mae500_stab_val/plots/ensemble_gain_vs_ssl_epoch.png)

Ensembling mean-pool + Q8 across the SSL-pretraining ladder (ep50→500): the
ensemble top-1 tracks backbone strength (47.1 % → 54.9 %), while the gain over the
best single member stays a modest, roughly flat ~+1 pp (range +0.6…+1.6 pp, no
anomalies after correcting one stale-checkpoint data artifact at ep300 — see
§6.3). The head/ensemble lever is small and stable; pretraining is the dominant
lever (~8 pp over the same range).

## 5. Validation on the test set (LB)

We submit each individual head and each ensemble to Kaggle under two inference
settings: **basic** (single centre forward) and **champion TTA** (3 scales
{0.857, 1.0, 1.143} × horizontal flip = 6 views, softmax-averaged). The val axis
is the basic OOF top-1 from §1–2.

![val vs LB scatter](../../outputs/ensemble/mae500_stab_val/plots/val_vs_lb_scatter.png)

*(2×2: rows = basic / champion-TTA LB, columns = public / private; grey ○ =
single head, red ★ = ensemble; dashed line is y = x.)*

### Full leaderboard table

LB scores are Kaggle top-1 (fraction); val is OOF top-1 (%).

| model | type | val % | pub basic | priv basic | pub TTA | priv TTA |
|-------|------|------:|----------:|-----------:|--------:|---------:|
| ens-seed-Q8×3 | ens | 55.63 | **0.5496** | 0.5619 | **0.5536** | 0.5636 |
| ens-all-axes | ens | 55.24 | 0.5432 | **0.5627** | 0.5449 | **0.5682** |
| ens-diverse4-s42 | ens | 55.95 | 0.5415 | 0.5613 | 0.5446 | 0.5665 |
| ens-arch-s42 | ens | 55.64 | 0.5365 | 0.5578 | 0.5438 | 0.5601 |
| perceiverQ8-s42 | single | 54.31 | 0.5345 | 0.5486 | 0.5409 | 0.5509 |
| perceiverQ8-s43 | single | 53.86 | 0.5325 | 0.5494 | 0.5377 | 0.5572 |
| perceiverQ8-s44 | single | 52.99 | 0.5365 | 0.5506 | 0.5409 | 0.5497 |
| perceiverQ16-s43 | single | 53.54 | 0.5322 | 0.5491 | 0.5371 | 0.5526 |
| DivSpaceTimeK9-s42 | single | 53.58 | 0.5290 | 0.5465 | 0.5261 | 0.5500 |
| meanpool-s42 | single | 53.33 | 0.5227 | 0.5384 | 0.5218 | 0.5373 |

### Ensemble gain (best ensemble − best single), all four LB variants

| LB variant | best single | best ensemble | gain |
|------------|-------------|---------------|------|
| public, basic | 0.5365 (Q8-s44) | 0.5496 (seed) | **+1.31 pp** |
| private, basic | 0.5506 (Q8-s44) | 0.5627 (all-axes) | **+1.21 pp** |
| public, TTA | 0.5409 (Q8-s42/s44) | 0.5536 (seed) | **+1.27 pp** |
| private, TTA | 0.5572 (Q8-s43) | 0.5682 (all-axes) | **+1.10 pp** |

**The val gain transfers to test under every setting** (+1.1…+1.3 pp); every
ensemble sits up-and-right of the single-head cluster in all four panels. Seed
diversity transfers at least as well as architecture diversity (basic public:
arch-only 0.5365 only ties the best single — its weak members drag it down; the
gap narrows but persists under TTA). **Best overall LB: 0.5536 public
(seed-Q8×3, TTA) and 0.5682 private (all-axes, TTA).**

### Effect of champion TTA

| inference | mean Δ public | mean Δ private |
|-----------|---------------|----------------|
| champion TTA vs basic (10 models) | **+0.29 pp** | **+0.30 pp** |

TTA is a **small and inconsistent** lift: it helps most models ~+0.3–0.8 pp but
*hurts* the two weakest heads on public (mean-pool −0.09, DivST-K9 −0.29) and is
near-zero/negative for a couple of cells on private (mean-pool −0.11, Q8-s44
−0.09). It does not change any ranking: ensembles still beat singles, seed ≈
architecture still holds. Per-model deltas (pp):

| model | Δ pub | Δ priv |
|-------|------:|-------:|
| ens-arch-s42 | +0.73 | +0.23 |
| perceiverQ8-s42 | +0.64 | +0.23 |
| perceiverQ8-s43 | +0.52 | +0.78 |
| perceiverQ16-s43 | +0.49 | +0.35 |
| perceiverQ8-s44 | +0.44 | −0.09 |
| ens-seed-Q8×3 | +0.40 | +0.17 |
| ens-diverse4-s42 | +0.31 | +0.52 |
| ens-all-axes | +0.17 | +0.55 |
| meanpool-s42 | −0.09 | −0.11 |
| DivSpaceTimeK9-s42 | −0.29 | +0.35 |

### Val→LB calibration (basic)

Public LB ≈ val − 0.7 pp; private LB ≈ val + 1 pp. The val→LB *ranking* holds at
the single-vs-ensemble level, though among ensembles the val ordering is not a
perfect predictor of LB ordering (e.g. diverse-4 has the best val but seed-Q8×3
the best public LB) — consistent with ~±1 pp LB sampling noise on N=6913.

## 6. Robustness investigations (settling the diversity claim)

Four follow-ups to decide whether the honest headline is "seed ≈ architecture"
or "diversity failed because of weak members". All offline on the val logits,
softmax OOF, with paired bootstrap CIs on the gain.

![investigations](../../outputs/ensemble/mae500_stab_val/plots/investigations.png)

### #1 — Fair diversity test (matched accuracy and matched count)

| member set | n | ens top-1 | gain | gain 95 % CI |
|------------|---|-----------|------|--------------|
| seed-only (Q8 × s42/s43/s44) | 3 | 55.63 | +1.32 | [+0.63, +1.96] |
| arch-matched (Q2/Q4/Q8 @ s42) | 3 | 55.51 | +1.20 | [+0.52, +1.88] |
| arch-weak (mp/Q8/DivST-K9 @ s42) | 3 | 55.64 | +1.33 | [+0.64, +2.07] |
| arch-3-seed (mp/Q8/DivST-K9 × 3) | 9 | 56.22 | +1.91 | [+1.20, +2.66] |
| all-axes (4 heads, 3 seeds) | 4 | 55.24 | +1.54 | [+0.84, +2.22] |

**No architecture set beats seed-only outside CI.** All three 3-member sets land
on top of each other (55.5–55.6, gains +1.2…+1.3, fully overlapping CIs) — even
the *matched-accuracy* arch set (Q2/Q4/Q8, all ~54 %) is not higher than seed. The
only set that gains more is the 9-member one, and it does so purely by **member
count**, not diversity type. This rules out the n=1 confound: **seed ≈
architecture diversity** is the honest statement.

### #2 — Are the weak heads dead weight?

| base set | ablation | top-1 | Δ vs full |
|----------|----------|-------|-----------|
| diverse-4 | full (mp+Q8+DivST+Q16) | 55.95 | — |
| | − mean-pool | 55.70 | **−0.25** |
| | − DivST-K9 | 55.64 | **−0.31** |
| | − both (Q8+Q16) | 55.23 | **−0.73** |
| arch-only | full (mp+Q8+DivST) | 55.64 | — |
| | − mean-pool | 55.08 | **−0.56** |
| | − DivST-K9 | 54.89 | **−0.76** |
| | − both (= Q8 single) | 54.31 | **−1.33** |

**The dominated heads are *not* dead weight.** Dropping mean-pool or DivST-K9
*lowers* the ensemble every time (−0.25 to −0.76 pp); they each contribute
positive diversity value despite being individually weaker. (This corrects an
earlier over-statement that decorrelation "adds no correct votes" — it does; it
just does not let the architecture set *beat* the seed set.)

### #3 — The ep300 dip was a stale-checkpoint pipeline artifact (corrected)

The −1.16 pp dip was **not a model effect but a data-pipeline bug**: the gymnote
fleet collection had grabbed a *mid-training* checkpoint of `perceiverQ8-mae300-s42`
(≈ epoch 22, 46.78 % val) instead of its final epoch-50 model, because the
collection ran before this run finished (it migrated lotte → rouget). The run
itself trained fine — there was no collapse. Re-fetching the **final** checkpoint
from the origin host (rouget; logged epoch-50 val 52.63 %, re-dump 52.71 %) and
recomputing, the ep300 ensemble is **54.28 % (gain +0.79 pp)** — on trend
(ep250 53.77 → ep300 54.28 → ep350 54.65). The corrected gain-vs-epoch curve in
§4 is smooth, with gain a flat +0.6…+1.6 pp at every epoch.

*Data-quality note.* The SSL-epoch **ladder** checkpoints come from a fleet
collection that, for a handful of runs, predated their final epoch; I re-verified
the cached val logits against the training logs and corrected the one materially
wrong member (Q8-mae300-s42, −5.9 pp). The **mae500** members used everywhere
else (§1–§5, §6.1–6.2, §6.4) re-dump to their logged final values (verified on
the origin hosts), so those results are unaffected.

### #4 — Val→LB transfer vs weak-member share

Residual = val − public-basic LB (positive ⇒ val over-states LB). Across the 10
submitted models, **r(weak-share, residual) = +0.50**; mean residual is +1.29 pp
for weak-heavy models (≥ 50 % mp/DivST members) vs +0.36 pp for lean ones — the
mean-pool/DivST-heavy ensembles **under-transfer on the public split**. *Caveat:*
on the **private** split these same weak-heavy ensembles (all-axes, diverse-4)
were the **best** (0.5682 / 0.5665), so the public under-transfer is most likely
public/private split sampling (~±1 pp) rather than a robust weakness.



1. **Ensembling adds a real but small ~+1.3 pp**, confirmed on the disjoint test
   set — an order of magnitude below the pretraining lever.
2. **The diversity axis is interchangeable**: seed ≈ architecture. The cheap
   Perceiver-Q8 ensembled across seeds is as good as mixing architectures.
3. **Diversity is axis-agnostic; gain scales with member count, not type**
   (§6.1): seed ≈ architecture even at matched accuracy. The weaker heads are
   *not* dead weight — they contribute positive value when included (§6.2,
   dropping them costs 0.25–0.76 pp) — but seed-diversity of the single best head
   captures the same ~+1.3 pp at the lowest cost.
4. **Plain softmax averaging is the right combiner**; learned weights do not beat
   it at this val size.
5. **Champion TTA is a minor, unreliable lever** (~+0.3 pp mean, sometimes
   negative for the weaker heads) and orthogonal to ensembling — it shifts every
   point up slightly but changes no ranking. Best LB overall: **0.5536 public /
   0.5682 private** (seed-Q8×3 / all-axes ensembles, TTA).
