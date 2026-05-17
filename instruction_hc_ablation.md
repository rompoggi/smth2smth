# HC / mHC ablation — launch instructions (Piranha)

**Spec:** [`experiments/mHC_HC.md`](experiments/mHC_HC.md)
**Configs:** `configs/experiment/track_a_hc_ablation_{baseline,shc,mhc}.yaml`
**Encoder base:** `checkpoints/track_a/ssl/sole_encoder.pt` — VideoMAE v2 ViT-B,
200 epochs SSL on SSv2 train+test frames, avg loss 0.4314.

All three arms fork from the same encoder so any difference is attributable to
the residual structure alone. 3 seeds per arm × 3 arms = 9 FT runs.

---

## Shared setup (run once on Piranha)

```bash
cd /path/to/smth2smth

export REPO_ROOT="$PWD"
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export BATCH_TAG="$(date +%Y%m%d)"

mkdir -p logs/hc checkpoints/track_a/hc

PY="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing .venv — run: uv sync" >&2
  exit 1
fi

# Confirm the base encoder is in place and is the ViT-B sole run.
test -s checkpoints/track_a/ssl/sole_encoder.pt || { echo "missing sole_encoder.pt"; exit 1; }

nvidia-smi
```

**Hard rules (all 9 runs):**

- Closed world: `model.init_from=checkpoints/track_a/ssl/sole_encoder.pt`, no other weights.
- Honest val: `dataset.use_official_val=true`, `dataset.include_val_in_train=false` (already in YAMLs).
- Seeds: 42, 43, 44 — **the same three seeds are reused across all three arms** (seed-paired comparison). Pass via Hydra `seed=N`.
- HC scalars (`alpha_pre`, `beta`, `M`, `M_raw`, `alpha_out`) automatically get `weight_decay=0` and skip layer-wise LR decay — see `_is_hc_scalar` in `pipelines/train.py`. Do NOT override `training.layer_decay` or attempt to re-enable wd on HC scalars.
- AMP: `precision=bf16` (already in champion train recipe). The mHC Sinkhorn-Knopp `exp`/normalize is forced to fp32 internally — do not change.
- ViT submit TTA: `[0.857, 1.0, 1.143]` + flip (already in YAMLs).

---

## Run order

| Arm | Hydra experiment                  | Residual | Wall (1 seed) | Wall (3 seeds) |
|-----|-----------------------------------|----------|---------------|----------------|
| (a) | `track_a_hc_ablation_baseline`    | Pre-Norm | ~3–4 h        | ~10–12 h       |
| (b) | `track_a_hc_ablation_shc`         | SHC n=4  | ~3.5–4.5 h    | ~11–13 h       |
| (c) | `track_a_hc_ablation_mhc`         | mHC n=4  | ~3.5–4.5 h    | ~11–13 h       |

**Sequencing rule** (from `experiments/mHC_HC.md` §7):
1. Run **(a)** for all 3 seeds first. Compute mean honest EMA val Top-1.
2. Run **(b)** for all 3 seeds. If `mean(b) < mean(a) − 0.3 pt`, **abort** — HC is dead at this scale, do not run (c).
3. Run **(c)** for all 3 seeds.

Total ~30 h after the encoder is in place. Fits roughly two overnight windows on a single 3090.

---

## Phase (a) — Pre-Norm baseline (3 seeds)

```bash
cd "$REPO_ROOT"
for SEED in 42 43 44; do
  LOG="logs/hc/piranha_hc_baseline_s${SEED}_${BATCH_TAG}.log"
  PID="logs/hc/piranha_hc_baseline_s${SEED}_${BATCH_TAG}.pid"

  nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
    "$PY" -u -m smth2smth.pipelines.train \
    experiment=track_a_hc_ablation_baseline track=a seed=${SEED} \
    > "$LOG" 2>&1 &
  echo $! > "$PID"
  echo "Started baseline seed=${SEED} pid=$(cat "$PID") log=$LOG"

  # Wait for this seed to finish before launching the next (single 3090).
  wait $(cat "$PID")
done
```

**Verify after each seed:**

```bash
test -s checkpoints/track_a/hc/baseline_s42_ft.pt && echo "s42 ok"
tail -n 30 logs/hc/piranha_hc_baseline_s42_${BATCH_TAG}.log | grep -E "val top1|ema val top1"
```

**Stability CSV:** `logs/hc/baseline_s${SEED}_stability.csv` — one row per optimizer step with `loss`, `grad_norm_global`, per-block grad norms, and (no-op for Pre-Norm) HC drift columns.

---

## Phase (b) — Static HC, n=4 (3 seeds)

Gate: confirm `mean(a)` first.

```bash
.venv/bin/python -c "
import torch, glob
for s in [42,43,44]:
    p = f'checkpoints/track_a/hc/baseline_s{s}_ft.pt'
    pl = torch.load(p, map_location='cpu', weights_only=False)
    print(s, pl.get('extra',{}).get('val_top1'))
"
```

Decision: if all three (a) seeds finished and the mean looks healthy (champion baseline is ~37 % top-1), launch (b). Otherwise diagnose first.

```bash
cd "$REPO_ROOT"
for SEED in 42 43 44; do
  LOG="logs/hc/piranha_hc_shc_s${SEED}_${BATCH_TAG}.log"
  PID="logs/hc/piranha_hc_shc_s${SEED}_${BATCH_TAG}.pid"

  nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
    "$PY" -u -m smth2smth.pipelines.train \
    experiment=track_a_hc_ablation_shc track=a seed=${SEED} \
    > "$LOG" 2>&1 &
  echo $! > "$PID"
  echo "Started shc seed=${SEED} pid=$(cat "$PID") log=$LOG"
  wait $(cat "$PID")
done
```

**Identity-init sanity at epoch 1:** the first few steps' train loss should track the Pre-Norm arm closely (HC's identity init means step 0 is byte-equal to Pre-Norm). Large divergence at step ≤ 100 indicates a wiring bug.

**Gate before (c):** abort if `mean(shc) < mean(baseline) − 0.3 pt`.

---

## Phase (c) — mHC, n=4, K=3 SK iters (3 seeds)

```bash
cd "$REPO_ROOT"
for SEED in 42 43 44; do
  LOG="logs/hc/piranha_hc_mhc_s${SEED}_${BATCH_TAG}.log"
  PID="logs/hc/piranha_hc_mhc_s${SEED}_${BATCH_TAG}.pid"

  nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
    "$PY" -u -m smth2smth.pipelines.train \
    experiment=track_a_hc_ablation_mhc track=a seed=${SEED} \
    > "$LOG" 2>&1 &
  echo $! > "$PID"
  echo "Started mhc seed=${SEED} pid=$(cat "$PID") log=$LOG"
  wait $(cat "$PID")
done
```

**SK sanity:** the per-step CSV's `sk_dev_*` columns should stay well below 1e-3 throughout training (3 iters at n=4 converges to ~0 deviation; we measured `row_dev_max=0.0` at init).

---

## Quick reference — Hydra experiment names

| Arm                     | Hydra `experiment=`                |
|-------------------------|------------------------------------|
| (a) Pre-Norm baseline   | `track_a_hc_ablation_baseline`     |
| (b) SHC n=4             | `track_a_hc_ablation_shc`          |
| (c) mHC n=4 SK=3        | `track_a_hc_ablation_mhc`          |

Always append: **`track=a seed={42,43,44}`**

---

## Optional: chained 9-run launcher

```bash
cd "$REPO_ROOT"
LOG="logs/hc/piranha_hc_chain_${BATCH_TAG}.log"
PID="logs/hc/piranha_hc_chain_${BATCH_TAG}.pid"

nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src bash -c '
  set -euo pipefail
  PY="'"$PY"'"
  for EXP in track_a_hc_ablation_baseline track_a_hc_ablation_shc track_a_hc_ablation_mhc; do
    for SEED in 42 43 44; do
      echo "[chain] $(date -Is) $EXP seed=$SEED"
      "$PY" -u -m smth2smth.pipelines.train experiment="$EXP" track=a seed=$SEED
    done
  done
  echo "[chain] $(date -Is) Done."
' > "$LOG" 2>&1 &
echo $! > "$PID"
```

This bypasses the (b)→(c) gate; only use it if you're committed to all 9 regardless of (b)'s result. Otherwise run the three phases manually.

---

## What lands on disk

Per-seed FT checkpoints:

```
checkpoints/track_a/hc/baseline_s{42,43,44}_ft.pt
checkpoints/track_a/hc/shc_n4_s{42,43,44}_ft.pt
checkpoints/track_a/hc/mhc_n4_sk3_s{42,43,44}_ft.pt
```

Per-seed stability CSVs (≈ 1 row per optimizer step × 60 epochs):

```
logs/hc/baseline_s{42,43,44}_stability.csv
logs/hc/shc_n4_s{42,43,44}_stability.csv
logs/hc/mhc_n4_sk3_s{42,43,44}_stability.csv
```

CSV columns: `step, epoch, lr, loss, grad_norm_global, grad_norm_encoder.blocks.0, ..., M_max_abs_encoder.blocks.0.attn_router, M_off_diag_mass_..., sk_dev_...` (mHC only for the last group).

---

## Plotting (after all 9 runs finish)

Mandatory plots per `experiments/mHC_HC.md` §9, mean ± std band across 3 seeds:

1. Train loss vs epoch (parsed from the per-epoch log lines).
2. Val Top-1 vs epoch (raw + EMA).
3. Val loss vs epoch.
4. Global grad-norm vs step (smoothed ~50 steps) — **the headline stability plot**.
5. Per-block grad-norm heatmap vs step.
6. HC mixing-matrix drift `||M − I||_F` vs epoch (arms b, c only).

A plotting script lives at `scripts/plot_hc_ablation.py` once you write it; until then the CSVs are pandas-friendly.

---

## Do not

- Run on a Sole encoder snapshot with <200 SSL epochs.
- Use DHC, or `hc_n > 4` — out of scope.
- Mix seeds across arms — seed-paired comparison is essential for the variance story.
- Apply weight decay to HC scalars (the train pipeline already enforces this; do not override).
- Forget that mHC's Sinkhorn block is fp32 under AMP bf16 — implemented in `sinkhorn_knopp()`, don't change.
- Launch (c) before (b)'s 3 seeds finish and meet the gate criterion.
