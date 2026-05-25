# Diverse classifier heads — Round 1 results (2026-05-25)

**Design brief:** [`diverse_classifier_heads_post_mae.md`](diverse_classifier_heads_post_mae.md)  
**Overnight launcher:** [`../RUN_diverse_heads.md`](../RUN_diverse_heads.md)  
**Shared recipe:** ep500 SSL encoder (`videomaev2_t4native_encoder_ep500.pt`), `tube_t=1`, `num_frames=4`, official val, LLRD 0.75 on backbone, new modules at full base LR 5e-4, 5-epoch global warmup, W&B project `smth2smth-diverse-heads`.

**Control reference (not re-run in this batch on all machines):** mean-pool head, same geometry — best prior run `mae450-ft-f4` val top1 **0.5349** (ep46); champion TTA LB **53.10–53.71%** depending on ckpt/submit. Round-1 diverse runs are compared against that bar, not against each other at peak before collapse.

---

## TL;DR

| Finding | Status |
|--------|--------|
| Projected +2–3.5 pt from Arch 3 (K=6) | **Not observed** — best live val 0.311, hard collapse ep7 |
| Arch 2 query collapse (brief §Arch 2 failure mode) | **Falsified** — query pairwise cosine stays ~0.09–0.13 through peak |
| Random-init temporal attention “poisons” backbone | **Falsified** — weight-diff: backbone spatial-attn norms move **~1–2%** vs SSL init at collapse |
| Actual failure mode | **Head MLP activation runaway** (`mlp_activity_ratio` → 50–100+), amplified by more random-init temporal blocks (monotonic in K) |
| Arch 2 Q=16 | **Highest upside** — peaked **EMA val 0.4503** (ep20) with healthy queries & action-localized attention; live model collapsed ep19 |
| Stabilized re-runs | Specified in [`diver_CL_head_continue.md`](diver_CL_head_continue.md) |

---

## Round 1 — completed / in-flight runs

| Run | Arch | K or Q | Status | Best live val top1 | Best EMA val top1 | Collapse epoch (live) | Peak `mlp_activity_ratio` |
|-----|------|--------|--------|-------------------|-------------------|----------------------|---------------------------|
| `arch2-perceiver-q16` | 2 Perceiver | Q=16 | **RUNNING** (ep27+, post-collapse) | 0.4331 (ep11) | **0.4503** (ep20) | **19** (val 0.062) | **100.55** (ep27) |
| `arch3-divided-st-k3` | 3 divided ST | K=3 | DONE (early stop ep18) | **0.3512** (ep4) | 0.343 (ep14) | gradual ep12–18 | 13.3 (ep15) |
| `arch3-divided-st-k6` | 3 divided ST | K=6 | DONE (early stop ep19) | **0.3110** (ep4) | 0.138 (ep7) | **7** (val 0.046) | **22.99** (ep9) |
| `arch3-divided-st-k12` | 3 divided ST | K=12 | STOPPED ep3 | 0.094 (ep1) | — | never trained | 2.8 (ep3) |
| `control-meanpool` | control | — | **Not launched** (assigned raie) | — | — | — | — |
| `arch1-attn-probe` | 1 | — | **Not launched** (piranha) | — | — | — | — |
| `arch4-aim` | 4 AIM | 12 blocks | **Not launched** (murene) | — | — | — | — |
| `arch3-divided-st-k9` | 3 K-sweep | K=9 | **Not in repo logs** | — | — | — | — |
| `arch2-perceiver-q32` | 2 Q-sweep | Q=32 | **Not launched** (sole) | — | — | — | — |

Logs: `logs/track_a/{RUN_NAME}_20260525.log`. Checkpoints: `checkpoints/track_a/videomaev2+ft/{RUN_NAME}.pt` (best; EMA when noted in log).

**K=12 post-mortem** (anguille): identity-at-init OK, but temporal modules in blocks 0–2 sit on LLRD-frozen spatial paths (~18–32× slower LR than new temporal tensors). Val regresses ep1→ep2; stopped ep3. **K=9** on same host reportedly trains (val ep1 0.196 → ep2 0.291) — confirms bottom-of-stack insertion is the structural hazard for K=12, distinct from the head-MLP runaway that dominates K=3/6/Q=16.

---

## 1. Collapse mechanism (negative result with evidence)

### 1.1 Hypothesis ruled out: backbone poisoning

The design brief and an early diagnostic hypothesis blamed **random-init temporal attention** for destabilizing pretrained spatial features. A **weight-diff** check (SSL init vs checkpoint at collapse) shows that **pretrained backbone spatial-attention weight norms change by only ~1–2%** through the collapse window. The MAE encoder representation is largely intact; the runaway is not “the backbone forgot SSL.”

### 1.2 Mechanism confirmed: head MLP positive feedback

All collapsing runs share one signature:

1. **Epochs 1–~10:** healthy training — val rises, `mlp_activity_ratio` ∈ [0.3, 4], `query_pairwise_cosine` ∈ [0.03, 0.13].
2. **Trigger:** one epoch (often one batch) pushes `mlp_activity_ratio` sharply upward (e.g. Q=16: 3.6 → **49.4** at ep19; K=6: 2.8 → **20.6** at ep7).
3. **Aftermath:** live val → ~0.05 (chance band), train loss → ~3.33, ratio stays **50–100+**; EMA decays slowly and preserves a usable snapshot **one epoch into** the cliff.

**Consolidated picture:** the **Perceiver head MLP** (~7M params, random init) sits on features whose distribution **drifts** as temporal blocks (Arch 3) or backbone co-adapt. At base LR 5e-4 with only 5-epoch warmup, the MLP enters a **positive-feedback loop**: larger activations → larger gradients → larger weights → larger activations. Temporal blocks are a **secondary amplifier** (more random-init modules ⇒ earlier / harder collapse), not the primary mover of backbone weights.

**Monotonic K ordering (Arch 3):**

| K | Temporal blocks | Collapse character | Best live val |
|---|-----------------|-------------------|---------------|
| 0 (Q=16 only) | 0 | Hard cliff ~ep19 | 0.43 live / **0.45 EMA** |
| 3 | last 3 | Slow bleed ep12–18 | 0.351 |
| 6 | last 6 | Hard cliff **ep7** | 0.311 |
| 12 | all 12 | **No learning** (LLRD clash) | 0.094 |

Single cause (head MLP runaway), monotonic effect in K — useful for a report subsection.

### 1.3 Why val early-stopping did not save the best model

Checkpoints track **live** val by default; EMA bests for Q=16 are **0.4377–0.4503** at ep18–20 while live val is already garbage. The saved `arch2-perceiver-q16.pt` after ep20 is the **EMA-best** checkpoint (log: “Saved new best checkpoint (ema)”). For K=6, best live 0.311 at ep4 is saved, but the interesting EMA trajectory peaks ep7 then dies — ensemble work should prefer **explicit EMA exports** pre-cliff.

### 1.4 Diagnostics already in the training loop

Per epoch (when `pool_head` exists):

```text
[head-diag] mlp_activity_ratio=..., query_pairwise_cosine=...
```

- **`mlp_activity_ratio`:** mean ‖MLP(norm2(z))‖ / ‖z‖ on queries (see `CrossAttnPoolHead` in `video_mae.py`). Brief’s Arch 1 “dead MLP” flag: **< 0.05**. Empirical collapse: **> 10** (often 50–100).
- **`query_pairwise_cosine`:** mean pairwise cosine of learnable queries. Brief’s collapse flag: **> 0.7**. Observed at peak: **0.09–0.13** — query collapse is **not** the failure mode for Q=16.

**Recommendation:** stop training when `mlp_activity_ratio > 10` (step-level logging if wired), not only val patience.

---

## 2. Architecture 2 — Perceiver Q=16 (partial success)

### 2.1 Training trajectory

- **Peak live val:** 0.4331 (ep11).
- **Peak EMA val:** **0.4503** (ep20) — still **below** control val ~0.53 but climbing pre-cliff.
- **Collapse:** ep19 live val 0.062, `mlp_activity_ratio` 49.4; ep20 EMA peak then live stuck ~0.05.
- **Queries:** cosine ~0.03 at init; **0.12** at ep11–18 — diverse, not collapsed.

### 2.2 Query-attention visualization (qualitative)

**Source checkpoint:** EMA-best weights at **ep20** (`mlp_activity_ratio` already 70 on live model — visualization is intentionally **not** the collapsed live weights).

**Figures:** `report/romain.poggi/figures/q16_query_attention.png`, `q16_query_attention_grid.png`.

**Observations (report must disclose EMA/pre-collapse snapshot):**

1. **Queries stay specialized** — distinct attention maps per query (pairwise cosine ~0.09 in weights; visually different spatial focus).
2. **Action localization** — diffuse on static early frames; concentrated on hand/object in motion-bearing frames (e.g. frame 4). Mean-pool averages **788** tokens including low-information frames; Perceiver **preserves** motion-bearing structure the control discards.
3. **Implication:** Arch 2 failed on **optimization**, not on the **architecture** assumed in UniFormerV2’s Q=1>Q=16 frozen-encoder result. Full-FT + strong MAE pretrain is a different regime.

**Counterfactual:** stabilized Q=16 that trains past step ~105k may exceed mean-pool LB — highest-EV follow-up (see continue doc).

---

## 3. Architecture 3 — divided space-time + Perceiver

### 3.1 K=6 (brochet)

- identity-at-init OK; encoder-missing=48 (expected).
- Best live **0.3110** ep4; cliff ep7 (`mlp_activity_ratio` 20.6).
- Recovered partially ep11–14 (live val ~0.09–0.13) but never beats ep4; early stop ep19.

### 3.2 K=3 (anguille)

- Best live **0.3512** ep4 — **better than K=6** at same recipe.
- Slower decline; `mlp_activity_ratio` 7–13 in late epochs — same mechanism, weaker amplifier.

### 3.3 K=12

- Did not learn: val **0.094 → 0.055** by ep2. Separate failure (LLRD vs full-LR temporal at bottom blocks), not MLP runaway alone.

### 3.4 vs brief projections

Brief expected **+2.0 to +3.5 pt** over 53.71% LB for K=6. Round 1 did not approach control on val or LB. Temporal modeling may still help **after** stabilization; T'=4 (tube_t=1) is less degenerate than the brief’s T'=2 assumption.

---

## 4. Architectures 1, 4, control — not yet measured

Overnight grid assigned **raie / piranha / murene** for control, Arch 1, Arch 4. No `logs/track_a/` entries on this repo snapshot. Treat as **pending**; do not infer failure/success.

---

## 5. Falsified or revised claims from the design brief

| Brief claim | Round-1 verdict |
|-------------|----------------|
| Arch 3 highest single-model EV (+2–3.5 pt) | **Not supported** without stabilization |
| Arch 2 query collapse at Q=16 on SSv2 | **Rejected** — cosine stays low; training dynamics fail |
| Separate param group at full LR for new modules sufficient | **Insufficient** for large random-init MLP + temporal blocks |
| identity-at-init sanity check | **Necessary and passed** on all Arch 3 runs — but does not prevent mid-training runaway |
| Mean-pool competitive because aggregation is optimal | **Open** — mean-pool may win despite suboptimal pooling if Perceiver cannot finish training |

---

## 6. Report narrative (three composable subsections)

1. **Collapse with mechanism** — weight-diff falsifies backbone poisoning; `mlp_activity_ratio` implicates head MLP; K-monotonicity.
2. **Perceiver queries action-localize under full-FT VideoMAE** — visualization from **EMA pre-collapse** checkpoint; contradicts frozen-encoder Q-sweep pessimism.
3. **Mean-pool vs stabilized Perceiver** — pending Run 1 in continue doc; headline either “LR-separated Perceiver matches/exceeds mean-pool on 4-frame MAE-FT SSv2” or “mean-pool surprisingly strong on strong backbones.”

---

## 7. Artifacts

| Artifact | Path |
|----------|------|
| Q=16 attention maps | `report/romain.poggi/figures/q16_query_attention*.png` |
| W&B project | `smth2smth-diverse-heads` |
| Best Q=16 ckpt (EMA) | `checkpoints/track_a/videomaev2+ft/arch2-perceiver-q16.pt` |
| Best K=3 / K=6 | `…/arch3-divided-st-k3.pt`, `…/arch3-divided-st-k6.pt` |

**Next:** [`diver_CL_head_continue.md`](diver_CL_head_continue.md) — stabilization recipe and run queue (machine assignment TBD).
