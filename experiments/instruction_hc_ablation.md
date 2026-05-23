# HC / mHC ablation — launch instructions

**Spec:** [`experiments/mHC_HC.md`](experiments/mHC_HC.md)
**Configs:** `configs/experiment/track_a_hc_ablation_{baseline,shc,mhc}.yaml`
**Encoder base:** `checkpoints/track_a/ssl/sole_encoder.pt` — VideoMAE v2 ViT-B,
200 epochs SSL on SSv2 train+test frames, avg loss 0.4314. On some machines this
file is named `e_r1_encoder.pt`; symlink before launch (see "Per-machine setup").

All three arms fork from the same encoder so any difference is attributable to
the residual structure alone:

- (a) baseline — standard Pre-Norm
- (b) shc — static hyper-connections, n=4
- (c) mhc — manifold-constrained hyper-connections, n=4, K=3 SK iters

3 seeds × 3 arms = 9 FT runs. The `baseline_s42` slot is grafted from a
parallel `e_r1_ft` run (same encoder, same recipe) to avoid duplicate work;
shc and mhc run all three seeds locally.

---

## Per-machine setup

```bash
cd /path/to/smth2smth

export REPO_ROOT="$PWD"
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export BATCH_TAG="$(date +%Y%m%d)"

mkdir -p logs/hc checkpoints/track_a/hc

# Python venv (MUST include the trainer.py mixup_cutmix_switch fix; otherwise
# every HC run crashes at training start with an "alpha must be positive"
# ValueError — the YAMLs use mixup_cutmix_switch with no outer alpha).
PY="${REPO_ROOT}/.venv/bin/python"
[[ -x "$PY" ]] || { echo "missing .venv — run: uv sync" >&2; exit 1; }

# SSv2 frames must be present at data/{train,val,test}.
test -d data/train && test -d data/val \
  || { echo "missing data/{train,val}"; exit 1; }

# Base encoder. If this machine has it as e_r1_encoder.pt, symlink first:
#   ln -sfn e_r1_encoder.pt checkpoints/track_a/ssl/sole_encoder.pt
test -s checkpoints/track_a/ssl/sole_encoder.pt \
  || { echo "missing sole_encoder.pt (symlink from e_r1_encoder.pt if needed)"; exit 1; }

nvidia-smi
```

---

## Hard rules (all 9 runs)

- Encoder: `model.init_from=checkpoints/track_a/ssl/sole_encoder.pt`, no other weights.
- Honest val: `dataset.use_official_val=true`, `dataset.include_val_in_train=false` (already in YAMLs).
- Recipe (already in YAMLs; do NOT override):
  - `model.dropout=0.0`, `model.gradient_checkpointing=true`
  - Effective batch = 64 in all three arms:
    - baseline: `batch_size=16, grad_accum_steps=4`
    - shc/mhc: `batch_size=4, grad_accum_steps=16` — HC's 4×-wider residual stream forces bs=4 on 24 GiB
  - `training.epochs=60`, `warmup_epochs=5`, `lr=5e-4`, `layer_decay=0.75`, `ema_decay=0.999`
  - AMP: `precision=bf16`. mHC Sinkhorn-Knopp `exp`/normalize is forced fp32 internally — do not change.
- HC scalars (`alpha_pre`, `beta`, `M`, `M_raw`, `alpha_out`) automatically get `weight_decay=0` and skip layer-wise LR decay (see `_is_hc_scalar` in `pipelines/train.py`).
- Seeds:
  - baseline arm — local: `{43, 44}`. s42 is grafted from `e_r1_ft.pt` (see "s42 graft").
  - shc and mhc arms — local: `{42, 43, 44}`.
  - **Seed-paired** across arms: the s42/s43/s44 trio is reused, the same encoder, the same recipe — only the residual variant changes.
- TTA at submit time: `tta_scales=[0.857, 1.0, 1.143]` + flip (already in YAMLs).
- Stability CSV: emitted every 10 optimizer steps (~4200 rows per seed for 60 epochs).

---

## Wall-time estimates (single 3090, ~24 GiB)

Measured on Piranha with the new recipe (gradient_checkpointing + eff_bs 64),
epoch 1 dataloader warmup included:

| Arm | Per-epoch wall | Per-seed wall (60 ep) | Per arm (3 seeds, sequential) |
|-----|----------------|-----------------------|-------------------------------|
| (a) baseline | ~25 min | ~24–28 h | ~48–56 h (s43 + s44 only) |
| (b) shc | ~25–35 min* | ~25–35 h* | ~75–105 h* |
| (c) mhc | ~25–35 min* | ~25–35 h* | ~75–105 h* |

*HC arms inherit baseline throughput plus modest overhead from the wider residual stream. Measure once at epoch 1 before committing.

Parallelizing one seed per machine across 2-3 boxes collapses an arm to ~1× per-seed wall.

---

## Run order

Sequencing rule (from [`experiments/mHC_HC.md`](experiments/mHC_HC.md) §7):

1. Finish baseline (a) at all three seeds (s42 graft + s43/s44 local). Compute mean honest EMA val Top-1.
2. Run shc (b) for all 3 seeds. If `mean(b) < mean(a) − 0.3 pt`, **abort** — HC is dead at this scale, do not run (c).
3. Run mhc (c) for all 3 seeds.

---

## Phase (a) — Pre-Norm baseline

### s42: graft from e_r1_ft

The `e_r1` champion FT (`track_a_ssl_finetune_e_r1`) at seed=42 uses the same
encoder, model architecture, and training recipe as this arm at seed=42. To
avoid duplicate work, rsync the e_r1 checkpoint into the HC arm's expected path
once it lands:

```bash
# After e_r1_ft completes on its host (looking for ~44%+ EMA val top1):
rsync -av <other-host>:.../checkpoints/track_a/ssl/e_r1_ft.pt \
  "$REPO_ROOT/checkpoints/track_a/hc/baseline_s42_ft.pt"
```

Note: `e_r1_ft` does NOT emit `logs/hc/baseline_s42_stability.csv`. The headline
grad-norm plot will be N=2 (s43, s44) at the s42-aligned column. The val Top-1
plot is unaffected.

### s43, s44: local runs

```bash
cd "$REPO_ROOT"
for SEED in 43 44; do
  LOG="logs/hc/thon_hc_baseline_s${SEED}_${BATCH_TAG}.log"
  PID="logs/hc/thon_hc_baseline_s${SEED}_${BATCH_TAG}.pid"

  nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
    "$PY" -u -m smth2smth.pipelines.train \
    experiment=track_a_hc_ablation_baseline track=a seed=${SEED} \
    > "$LOG" 2>&1 &
  echo $! > "$PID"
  echo "Started baseline seed=${SEED} pid=$(cat "$PID") log=$LOG"

  wait $(cat "$PID")  # single-GPU: serialize seeds
done
```

For multi-machine parallel runs, drop the `wait` and assign one seed per box. See "Parallel launch".

### Verify after each seed

```bash
for SEED in 42 43 44; do
  test -s checkpoints/track_a/hc/baseline_s${SEED}_ft.pt && echo "s${SEED} ckpt ok"
done
tail -n 30 logs/hc/thon_hc_baseline_s43_${BATCH_TAG}.log | grep -E "val top1|ema val top1"
```

Stability CSV: `logs/hc/baseline_s{43,44}_stability.csv` — `grad_norm_global` + per-block + (no-op for Pre-Norm) HC drift columns.

---

## Phase (b) — Static HC, n=4

Gate: confirm `mean(a)` first.

```bash
.venv/bin/python -c "
import torch
vals = []
for s in [42, 43, 44]:
    pl = torch.load(f'checkpoints/track_a/hc/baseline_s{s}_ft.pt',
                    map_location='cpu', weights_only=False)
    v = pl.get('extra', {}).get('val_top1')
    vals.append(v); print(s, v)
print('mean', sum(vals)/len(vals))
"
```

Decision: if the mean is in the e_r1-equivalent range (champion ~44%+), launch (b). Otherwise diagnose first.

```bash
cd "$REPO_ROOT"
for SEED in 42 43 44; do
  LOG="logs/hc/thon_hc_shc_s${SEED}_${BATCH_TAG}.log"
  PID="logs/hc/thon_hc_shc_s${SEED}_${BATCH_TAG}.pid"

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

## Phase (c) — mHC, n=4, K=3 SK iters

```bash
cd "$REPO_ROOT"
for SEED in 42 43 44; do
  LOG="logs/hc/thon_hc_mhc_s${SEED}_${BATCH_TAG}.log"
  PID="logs/hc/thon_hc_mhc_s${SEED}_${BATCH_TAG}.pid"

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

## Parallel launch (one seed per machine)

Each per-seed run is independent (different `checkpoint_path`, different
`stability_log_path`). To fan a phase across boxes, drop the `wait` and assign
one seed per machine. Example for baseline_s44 on machine B:

```bash
cd "$REPO_ROOT"
SEED=44
LOG="logs/hc/thon_hc_baseline_s${SEED}_${BATCH_TAG}.log"
PID="logs/hc/thon_hc_baseline_s${SEED}_${BATCH_TAG}.pid"
nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$PY" -u -m smth2smth.pipelines.train \
  experiment=track_a_hc_ablation_baseline track=a seed=${SEED} \
  > "$LOG" 2>&1 &
echo $! > "$PID"
```

When all machines for a phase finish, rsync `checkpoints/track_a/hc/*_ft.pt`
and `logs/hc/*_stability.csv` to one host before evaluating the gate or
plotting.

---

## Quick reference — Hydra experiment names

| Arm                     | Hydra `experiment=`                |
|-------------------------|------------------------------------|
| (a) Pre-Norm baseline   | `track_a_hc_ablation_baseline`     |
| (b) SHC n=4             | `track_a_hc_ablation_shc`          |
| (c) mHC n=4 SK=3        | `track_a_hc_ablation_mhc`          |

Always append `track=a seed=<N>`. For baseline, s42 is graft-only.

---

## What lands on disk

Per-seed FT checkpoints:

```
checkpoints/track_a/hc/baseline_s42_ft.pt        # rsynced from e_r1_ft (no local run)
checkpoints/track_a/hc/baseline_s{43,44}_ft.pt   # local
checkpoints/track_a/hc/shc_n4_s{42,43,44}_ft.pt
checkpoints/track_a/hc/mhc_n4_sk3_s{42,43,44}_ft.pt
```

Per-seed stability CSVs (every 10 optimizer steps × ~700 opt-steps/epoch × 60 epochs ≈ 4200 rows):

```
logs/hc/baseline_s{43,44}_stability.csv          # s42: not produced (grafted)
logs/hc/shc_n4_s{42,43,44}_stability.csv
logs/hc/mhc_n4_sk3_s{42,43,44}_stability.csv
```

CSV columns: `step, epoch, lr, loss, grad_norm_global, grad_norm_encoder.blocks.0, ..., M_max_abs_encoder.blocks.0.attn_router, M_off_diag_mass_..., sk_dev_...` (mHC only for the last group).

---

## Plotting (after all 9 runs finish)

Mandatory plots per [`experiments/mHC_HC.md`](experiments/mHC_HC.md) §9, mean ± std band across 3 seeds:

1. Train loss vs epoch (parsed from the per-epoch log lines).
2. Val Top-1 vs epoch (raw + EMA).
3. Val loss vs epoch.
4. Global grad-norm vs step (smoothed ~50 steps) — **the headline stability plot**. Baseline at s42 is N=2 here.
5. Per-block grad-norm heatmap vs step.
6. HC mixing-matrix drift `||M − I||_F` vs epoch (arms b, c only).

Until `scripts/plot_hc_ablation.py` exists, the CSVs are pandas-friendly.

---

## Do not

- Run on a Sole encoder snapshot with <200 SSL epochs.
- Use DHC, or `hc_n > 4` — out of scope.
- Mix seeds across arms — seed-paired comparison is essential for the variance story.
- Apply weight decay to HC scalars (the train pipeline already enforces this; do not override).
- Forget that mHC's Sinkhorn block is fp32 under AMP bf16 — implemented in `sinkhorn_knopp()`, don't change.
- Launch (c) before (b)'s 3 seeds finish and meet the gate criterion.
- Locally train `baseline_s42` — graft it from `e_r1_ft.pt`.
- Launch any of this without the `trainer.py` `mixup_cutmix_switch` fix on the local checkout — every run will crash at training start.
