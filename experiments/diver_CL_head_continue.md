# Diverse classifier heads — Round 2 (stabilized) run plan

**Prerequisites:** read [`diverse_CL_head_results.md`](diverse_CL_head_results.md) (collapse mechanism).  
**Design brief:** [`diverse_classifier_heads_post_mae.md`](diverse_classifier_heads_post_mae.md).  
**Launcher template:** adapt from [`../RUN_diverse_heads.md`](../RUN_diverse_heads.md).

**Goal (~4 days left):** (1) push single-model / ensemble above **53.7% LB**; (2) lock negative result with mechanism + optional stabilized positive for Perceiver / Arch 3.

---

## 0. Machine assignment (Round 2 — 2026-05-25, revised)

**Launcher:** [`../RUN_diverse_heads_stab.md`](../RUN_diverse_heads_stab.md)

**Hosts in use elsewhere:** `brochet`, `anguille` (~10 GB VRAM). **Control skipped:** ep500 mean-pool already trained.

| Host | `RUN_NAME` | `EXPERIMENT` | Notes |
|------|------------|--------------|-------|
| **baudroie** | `arch2-perceiver-q16-stab` | `track_a_diverse_arch2_perceiver_stab` | Lowest VRAM (~93M); ~4 GB already used |
| **ablette** | `arch3-divided-st-k3-stab` | `track_a_diverse_arch3_divided_st_k3_stab` | ~102M |
| **barbeau** | `arch3-divided-st-k6-stab` | `track_a_diverse_arch3_divided_st_stab` | ~111M; needs most free VRAM |
| **gymnote** | `arch2-perceiver-q16-stab-s42` | `track_a_diverse_arch2_perceiver_stab_s42` | Seed replica; CLI `seed=123` |

| Priority | `RUN_NAME` | Host |
|----------|------------|------|
| **1** | `arch2-perceiver-q16-stab` | baudroie |
| **2** | `arch3-divided-st-k6-stab` | barbeau |
| **3** | `arch3-divided-st-k3-stab` | ablette |
| **4** | `arch2-perceiver-q16-stab-s42` | gymnote |

Deferred: `control-meanpool-ep500`, `arch2-perceiver-q32`, Arch 1/4 Round 1.

**W&B (implemented):** `head/mlp_activity_ratio`, `head/query_pairwise_cosine`, `optim/lr/{group}`, `optim/warmup_lr_max/{group}`, `optim/warmup_epochs/{group}` each epoch.

---

## 1. Code / config work before any launch

Round 1 used **one global LR 5e-4** with LLRD routing `pool_head.*` and temporal params to the top LLRD group — still too aggressive for the ~7M random-init head MLP.

### 1.1 Trainer changes (`train.py` + YAML) — **implemented**

**Three optimizer groups** for `video_mae_vit` when `training.new_module_lr` is set:

| Group | Parameters | LR | Warmup epochs | `layer_decay` |
|-------|------------|-----|---------------|---------------|
| **backbone** | pretrained encoder blocks (LLRD as today) | 5e-4 × LLRD scale | 5 | 0.75 |
| **new_temporal** | `_is_new_temporal_param` (Arch 3/4) | **1e-4** | **10** | 1.0 (no decay) |
| **pool_head** | `pool_head.*`, `classifier.*` | **1e-4** | **10** | 1.0 |

Config group: `configs/train/videomae_official_ssv2_stab.yaml` (composed by `*_stab` experiment YAMLs).

Experiment presets:

- `track_a_diverse_arch2_perceiver_stab.yaml`
- `track_a_diverse_arch3_divided_st_stab.yaml`
- `track_a_videomae_control_ep500_f4.yaml` (control — no stab train group)

### 1.3 W&B

Keep project **`smth2smth-diverse-heads`**; tag runs `stab`, `round2`.

---

## 2. Stabilization recipe (all Round 2 runs)

**Shared overrides** (in addition to architecture-specific `EXTRA`):

```text
model.init_from=checkpoints/track_a/ssl/videomaev2_t4native_encoder_ep500.pt
training.new_module_lr=1e-4
training.new_module_warmup_epochs=10
training.max_grad_norm=1.0
training.stop_on_mlp_activity_ratio=10.0
training.wandb.project=smth2smth-diverse-heads
```

**Unchanged from Round 1:** batch 8 × grad_accum 8, 50 epochs, cosine, EMA 0.9999, official val, augment recipe, `log_interval_steps=25`.

**Do not change:** `layer_decay=0.75` on backbone; identity-at-init for Arch 3/4.

---

## 3. Runs (priority order)

### Run 1 — `arch2-perceiver-q16-stab` (highest priority)

| Field | Value |
|-------|--------|
| Architecture | Arch 2, Q=16, no temporal blocks |
| Why | Peaked **EMA val 0.4503** pre-collapse; queries healthy; action-localized attention |
| `EXPERIMENT` | `track_a_diverse_arch2_perceiver` |
| `EXTRA` | §2 shared only |
| `training.checkpoint_path` | `checkpoints/track_a/videomaev2+ft/arch2-perceiver-q16-stab.pt` |
| `training.wandb.name` | `arch2-perceiver-q16-stab` |
| Success criterion | Train past former cliff (~ep20+); **val top1 ≥ 0.50** live or EMA; no `mlp_activity_ratio` spike > 10 |
| If success | LB submit with champion TTA; enqueue **Run 4** seed replica immediately |

### Run 2 — `arch3-divided-st-k6-stab`

| Field | Value |
|-------|--------|
| Architecture | Arch 3, K=6 + Q=16 Perceiver |
| Why | Highest brief ceiling if stabilization works |
| `EXPERIMENT` | `track_a_diverse_arch3_divided_st` |
| `EXTRA` | §2 shared (default `temporal_layers=6`) |
| Success criterion | No ep7 cliff; val top1 **> 0.35** sustained past ep15; compare to K=3 stab |
| Failure interpretation | Temporal blocks structurally incompatible at this LR recipe even when head is stable — still a valid negative result |

### Run 3 — `arch3-divided-st-k3-stab`

| Field | Value |
|-------|--------|
| Architecture | Arch 3, K=3 |
| `EXTRA` | `model.temporal_layers=3` + §2 shared |
| Why | Fewer random-init temporal modules — if K=6 stab fails but K=3 succeeds, report **how many temporal blocks a converged MAE backbone can absorb** |

### Run 4 — `arch2-perceiver-q16-stab-s42` (conditional)

| Field | Value |
|-------|--------|
| Trigger | Run 1 reaches ≥ 0.50 val and stable `mlp_activity_ratio` |
| Change | `seed=123` (or `1`) only; same `-stab` recipe |
| Why | Ensemble diversity for Caruana stack with control + mean-pool |

### Run 5 — `control-meanpool-ep500`

| Field | Value |
|-------|--------|
| Architecture | mean-pool linear head |
| `EXPERIMENT` | `track_a_videomae_official_ssv2_ft` |
| `EXTRA` | `model.tube_t=1 dataset.num_frames=4` (fair T=4 geometry) |
| SSL | ep500 encoder (same as diverse runs) |
| LR note | Control head is tiny (~25k params) — **may keep base 5e-4** on head only, or single group; do **not** apply `new_module_lr` to backbone |
| Why | Refresh baseline so ensemble members share **same SSL epoch and frame count** |

---

## 4. Launch snippet (after §1 implemented)

```bash
cd /Data/romain.poggi/smth2smth
UV=/users/eleves-a/2021/romain.poggi/.local/bin/uv   # full path for nohup

# ╔════════════ EDIT ════════════╗
RUN_NAME=arch2-perceiver-q16-stab
EXPERIMENT=track_a_diverse_arch2_perceiver
EXTRA="training.new_module_lr=1e-4 training.new_module_warmup_epochs=10 training.max_grad_norm=1.0 training.stop_on_mlp_activity_ratio=10.0"
# ╚══════════════════════════════╝

SSL=checkpoints/track_a/ssl/videomaev2_t4native_encoder_ep500.pt
TAG=$(date +%Y%m%d)
LOG=logs/track_a/${RUN_NAME}_${TAG}.log
PID=logs/track_a/${RUN_NAME}_${TAG}.pid
CKPT=checkpoints/track_a/videomaev2+ft/${RUN_NAME}.pt
mkdir -p logs/track_a checkpoints/track_a/videomaev2+ft

{
  echo "# run: ${RUN_NAME}"
  echo "# started: $(date -Iseconds)"
  echo "# track: a"
  echo "# experiment_doc: experiments/diver_CL_head_continue.md"
  echo "# hydra: experiment=${EXPERIMENT} ${EXTRA}"
} > "$LOG"

PYTHONPATH=src PYTHONUNBUFFERED=1 nohup "$UV" run python -u -m smth2smth.pipelines.train \
  experiment=${EXPERIMENT} \
  model.init_from=${SSL} \
  training.checkpoint_path=${CKPT} \
  training.wandb.name=${RUN_NAME} \
  training.wandb.project=smth2smth-diverse-heads \
  ${EXTRA} >> "$LOG" 2>&1 &

echo $! > "$PID"
echo "launched ${RUN_NAME} pid=$(cat $PID) -> ${LOG}"
```

**Health check (~90 s):** same as Round 1 — step lines, no traceback; for Arch 3, `identity-at-init OK`; watch first `mlp_activity_ratio` in epoch 1 (expect < 2).

---

## 5. Monitoring checklist (per run)

| Signal | OK | Stop / investigate |
|--------|-----|-------------------|
| `mlp_activity_ratio` | < 4 through ep15 | **> 10** any epoch |
| `query_pairwise_cosine` | < 0.2 | > 0.7 |
| Live val | rising or flat | → 0.05 sudden |
| EMA val | tracks live pre-cliff | save EMA ckpt when live cliffs |
| `[diverse-arch] identity-at-init` | OK (Arch 3/4) | BROKEN |

---

## 6. Out of scope for this batch (unless a machine is idle)

| Item | Reason |
|------|--------|
| Arch 1 / Arch 4 Round 1 | Still unrun on grid — lower priority than stab Q=16 / K=6 |
| `arch2-perceiver-q32` | Q-sweep after Q=16 stab succeeds |
| `arch3-divided-st-k9` / `k12` | K=12 failed for LLRD reasons; K=9 only if K=6 stab works |
| LB submit for collapsed Round-1 ckpts | Misleading — use EMA ep18–20 only for analysis, not submit |

---

## 7. After Round 2

1. **Submit** best stabilized single model + champion TTA.  
2. **Caruana** stack: control-ep500 + Q=16-stab (+ seed replica) + any Arch 3 stab winner.  
3. **Update** [`diverse_CL_head_results.md`](diverse_CL_head_results.md) with Round 2 table (do not edit the original design brief).  
4. **Report subsections** — paste from results doc §6; include visualization caveat (EMA checkpoint).

---

**Status:** code + configs ready; dry-run with `./scripts/dryrun_diverse_heads_stab.sh` before overnight launch per [`RUN_diverse_heads_stab.md`](../RUN_diverse_heads_stab.md).
