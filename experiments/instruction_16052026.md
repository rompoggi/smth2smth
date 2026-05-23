# Track A overnight batch — per-machine launch instructions (2026-05-16)

**Purpose:** copy the **Machine prompt** block for your host into Cursor (or run the shell commands yourself).  
**Do not run training from an agent sandbox** — GPUs are on the VMs; only paste/execute these commands **on the target machine** in a real shell.

**Spec:** [`experiment_16052026.md`](experiment_16052026.md)  
**Configs:** `configs/experiment/track_a_ssl_pretrain_e{N}.yaml`, `track_a_ssl_finetune_e{N}.yaml`, `track_a_e4_tsm_sgdr.yaml` — see [quick reference](#quick-reference--hydra-experiment-names). Checkpoints: `checkpoints/track_a/ssl/e{N}_encoder.pt`, `e{N}_ft.pt` (E4 under `checkpoints/track_a/e4_ft.pt`).

---

## Machine ↔ experiment assignment

| Machine   | Experiment | Pipeline                                      | ~Wall   |
|-----------|------------|-----------------------------------------------|---------|
| **Anchois**   | E1         | MAE 100 ep → champion FT 60 ep                | ~11.5 h |
| **Ablette**   | E2         | MAE 150 ep (aug) → champion FT 60 ep          | ~13–14 h |
| **Sole**      | E3         | MAE 80 ep T=8 → champion FT 50 ep T=8         | ~13 h   |
| **Truite**    | E4         | R50+TSM supervised SGDR 90 ep (+ snapshots)   | ≤14 h   |
| **thon**      | E5         | MAE 100 ep → FT 50 ep → cRT 10 ep             | ~12 h   |
| **Roussette** | E6         | MAE 100 ep @224 → FT 50 ep @256               | ~12 h   |
| **Raie**      | E7         | MAE 150 ep (mask schedule) → champion FT 60 ep | ~13–14 h |

**Hard rules (all machines):**

- Closed world only — no external weights, no other VM’s checkpoints.
- Honest val: `dataset.use_official_val=true`, `dataset.include_val_in_train=false` (already in YAMLs).
- ViT submit TTA scales: **`[0.857, 1.0, 1.143]`** only (not `0.875/1.125` at base 224).
- CNN (E4) TTA scales: **`[0.875, 1.0, 1.125]`**.
- SSL pretrain data: **train + test** frames only (val excluded unless you explicitly override `pretrain.include_val_in_pretrain=true` — **do not** for this batch).

---

## Shared setup (every machine, once)

Run from the **repository root** (`smth2smth/`, where `configs/` and `src/` live).

```bash
cd /path/to/smth2smth    # <-- set to your clone on this VM

export REPO_ROOT="$PWD"
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export BATCH_TAG="20260516"   # date stamp for log/pid names

mkdir -p logs checkpoints/track_a/ssl checkpoints/track_a

# Prefer project venv (nohup often has no `uv` on PATH)
PY="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing .venv — run: uv sync" >&2
  exit 1
fi

# Optional: confirm GPU
nvidia-smi

# Optional: override dataset root (if data is not at ${REPO_ROOT}/data)
# export DATASET_ROOT=/your/path/to/data
# Then append to every launch:  dataset.root=$DATASET_ROOT
```

**Unbuffered logs + timestamps**

- `PYTHONUNBUFFERED=1` and `python -u` → lines appear immediately in log files.
- Training code prints **ISO-8601** timestamps on epoch **1**, **last**, and every **50** epochs, e.g.  
  `[videomae] 2026-05-16T23:01:02 epoch 50/100 avg loss ...`  
  `[2026-05-16T23:01:02] Epoch 50/60 | ...`

**After each `nohup` launch**

```bash
# Replace LOG and PID with the paths from that machine’s section
tail -n 40 "$LOG"
kill -0 "$(cat "$PID")" && echo "still running pid=$(cat "$PID")"
pgrep -af "smth2smth.pipelines.pretrain_videomae|smth2smth.pipelines.train"
```

**Wait until a phase finishes** before starting the next (unless noted otherwise):

```bash
# Encoder must exist and be non-empty (path from your machine’s checkpoint list below)
test -s checkpoints/track_a/ssl/ENCODER.pt && echo "encoder OK"

# FT best checkpoint (stage 1 / single-phase FT)
test -s checkpoints/track_a/ssl/FT.pt && echo "ft OK"
```

---

# Machine: Anchois — E1 (SSL champion control)

## Machine prompt (paste on Anchois)

You are on VM **Anchois** for experiment **E1**: VideoMAE ViT-S MAE pretrain (100 ep, minimal aug) then champion fine-tune (60 ep). This is the **control** for the batch. Do not change hyperparameters overnight. Use honest official val only. Launch jobs with `nohup` on the GPU host (not in a sandbox). After each phase, verify the log shows Hydra YAML then training lines with ISO timestamps.

**Checkpoints (this VM only):**

- `checkpoints/track_a/ssl/e1_encoder.pt`
- `checkpoints/track_a/ssl/e1_ft.pt` (+ `e1_ft.last.pt` resume artifact)

**Logs / PIDs:**

- `logs/anchois_e1_mae_pretrain_${BATCH_TAG}.log`
- `logs/anchois_e1_ft_champion_${BATCH_TAG}.log`

### Phase 1 — MAE pretrain (~7 h)

```bash
cd "$REPO_ROOT"
LOG="logs/anchois_e1_mae_pretrain_${BATCH_TAG}.log"
PID="logs/anchois_e1_mae_pretrain_${BATCH_TAG}.pid"

nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$PY" -u -m smth2smth.pipelines.pretrain_videomae \
  experiment=track_a_ssl_pretrain_e1 track=a \
  > "$LOG" 2>&1 &
echo $! > "$PID"
echo "Started MAE pretrain pid=$(cat "$PID") log=$LOG"
```

**Wait for:** log line `wrote trunk checkpoint` / `e1_encoder.pt`; `test -s checkpoints/track_a/ssl/e1_encoder.pt`.

### Phase 2 — Champion FT (~4.5 h)

```bash
cd "$REPO_ROOT"
LOG="logs/anchois_e1_ft_champion_${BATCH_TAG}.log"
PID="logs/anchois_e1_ft_champion_${BATCH_TAG}.pid"

nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$PY" -u -m smth2smth.pipelines.train \
  experiment=track_a_ssl_finetune_e1 track=a \
  > "$LOG" 2>&1 &
echo $! > "$PID"
echo "Started FT pid=$(cat "$PID") log=$LOG"
```

**Success note:** honest EMA val Top-1 **≥ 37.0%** by epoch 60. Submit later with ViT TTA `[0.857,1.0,1.143]` + flip (already in finetune YAML).

---

# Machine: Ablette — E2 (augmented MAE pretrain)

## Machine prompt (paste on Ablette)

You are on VM **Ablette** for **E2**: MAE pretrain **150 epochs** with flip + color jitter + grayscale (no RandAugment at pretrain), then champion FT 60 ep. **Start early** (~13–14 h total). Same FT recipe as E1; only pretrain aug differs. `nohup` only on GPU host.

**Checkpoints:** `e2_encoder.pt`, `e2_ft.pt`  
**Logs:** `logs/ablette_e2_mae_pretrain_${BATCH_TAG}.log`, `logs/ablette_e2_ft_champion_${BATCH_TAG}.log`

### Phase 1 — MAE pretrain (~8–9 h)

```bash
cd "$REPO_ROOT"
LOG="logs/ablette_e2_mae_pretrain_${BATCH_TAG}.log"
PID="logs/ablette_e2_mae_pretrain_${BATCH_TAG}.pid"

nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$PY" -u -m smth2smth.pipelines.pretrain_videomae \
  experiment=track_a_ssl_pretrain_e2 track=a \
  > "$LOG" 2>&1 &
echo $! > "$PID"
```

### Phase 2 — Champion FT (after encoder exists)

```bash
cd "$REPO_ROOT"
LOG="logs/ablette_e2_ft_champion_${BATCH_TAG}.log"
PID="logs/ablette_e2_ft_champion_${BATCH_TAG}.pid"

nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$PY" -u -m smth2smth.pipelines.train \
  experiment=track_a_ssl_finetune_e2 track=a \
  > "$LOG" 2>&1 &
echo $! > "$PID"
```

---

# Machine: Sole — E3 (T=8 end-to-end)

## Machine prompt (paste on Sole)

You are on VM **Sole** for **E3**: MAE + FT at **`num_frames=8`** (pretrain 80 ep, FT 50 ep), gradient checkpointing + `grad_accum_steps=2`. **Start early** (~13 h). Never change T between pretrain, FT, and submit. `nohup` on GPU host only.

**Checkpoints:** `e3_encoder.pt`, `e3_ft.pt`  
**Logs:** `logs/sole_e3_mae_pretrain_t8_${BATCH_TAG}.log`, `logs/sole_e3_ft_champion_t8_${BATCH_TAG}.log`

### Phase 1 — MAE pretrain T=8 (~8 h)

```bash
cd "$REPO_ROOT"
LOG="logs/sole_e3_mae_pretrain_t8_${BATCH_TAG}.log"
PID="logs/sole_e3_mae_pretrain_t8_${BATCH_TAG}.pid"

nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$PY" -u -m smth2smth.pipelines.pretrain_videomae \
  experiment=track_a_ssl_pretrain_e3 track=a \
  > "$LOG" 2>&1 &
echo $! > "$PID"
```

### Phase 2 — Champion FT T=8 (~5 h)

```bash
cd "$REPO_ROOT"
LOG="logs/sole_e3_ft_champion_t8_${BATCH_TAG}.log"
PID="logs/sole_e3_ft_champion_t8_${BATCH_TAG}.pid"

nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$PY" -u -m smth2smth.pipelines.train \
  experiment=track_a_ssl_finetune_e3 track=a \
  > "$LOG" 2>&1 &
echo $! > "$PID"
```

---

# Machine: Truite — E4 (CNN snapshot ensemble)

## Machine prompt (paste on Truite)

You are on VM **Truite** for **E4**: **supervised only** — ResNet-50 + TSM from scratch, **90 epochs**, SGDR (`T_0=30`), snapshots at cycles 1–3. No SSL pretrain. Independent of the live 150-ep Phase-2 run elsewhere. If epoch time > ~9 min and total wall > 14 h, stop and relaunch with `training.epochs=72` (3×24) — note in log. `nohup` on GPU host.

**Checkpoints:**

- `checkpoints/track_a/e4_ft.pt` (EMA best)
- `checkpoints/track_a/e4_ft_snap1.pt`, `_snap2.pt`, `_snap3.pt`

**Log:** `logs/truite_e4_supervised_sgdr90_${BATCH_TAG}.log`

### Single phase — supervised train (~overnight)

```bash
cd "$REPO_ROOT"
LOG="logs/truite_e4_supervised_sgdr90_${BATCH_TAG}.log"
PID="logs/truite_e4_supervised_sgdr90_${BATCH_TAG}.pid"

nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$PY" -u -m smth2smth.pipelines.train \
  experiment=track_a_e4_tsm_sgdr track=a \
  > "$LOG" 2>&1 &
echo $! > "$PID"
echo "Started SGDR train pid=$(cat "$PID") log=$LOG"
```

**Monitor for:** `[sgdr] saved cycle-N snapshot` lines at epochs 30, 60, 90.

### Phase 2 — continuation 90→150 ep (after phase 1 finishes)

**Wait for** `test -s checkpoints/track_a/e4_ft.last.pt` and log line `Done.` (or epoch 90/90 + cycle-3 snapshot). GPU headroom (~5.5 GiB / 24 GiB on RTX 3090) already supports **bs=16** (same as phase 1).

```bash
cd "$REPO_ROOT"
LOG="logs/truite_e4_continue150_${BATCH_TAG}.log"
PID="logs/truite_e4_continue150_${BATCH_TAG}.pid"

nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$PY" -u -m smth2smth.pipelines.train \
  experiment=track_a_e4_tsm_continue150 track=a \
  training.resume_from="${REPO_ROOT}/checkpoints/track_a/e4_ft.last.pt" \
  > "$LOG" 2>&1 &
echo $! > "$PID"
echo "Started E4 continue-150 pid=$(cat "$PID") log=$LOG"
```

**Schedule:** 10-ep linear warmup from epoch 91, then cosine over epochs 101–150 (`T_max=50`). Preset: `track_a_e4_tsm_continue150`. Best/last: `e4_ft.pt`, `e4_ft_150.last.pt`.

**Mid-run resume** (after crash/kill; do not use `e4_ft.pt` unless rolling back):

```bash
training.resume_from="${REPO_ROOT}/checkpoints/track_a/e4_ft_150.last.pt" \
  training.resume_apply_cfg_lr=false \
  training.warmup_start_epoch=90
```

Append to the same `truite_e4_continue150_${BATCH_TAG}.log`.

**Auto-chain (phase 1 still running):** watcher polls the phase-1 PID and launches phase 2 when it exits (no `kill`):

```bash
cd "$REPO_ROOT"
export BATCH_TAG=20260516
nohup env BATCH_TAG="$BATCH_TAG" bash scripts/truite_e4_chain_continue150.sh \
  >> "logs/truite_e4_chain_continue150_${BATCH_TAG}.log" 2>&1 &
echo $! > "logs/truite_e4_chain_continue150_${BATCH_TAG}.pid"
```

### Phase 3 — train+val fine-tune (after continue-150)

Low-LR polish: **train + official val** (`include_val_in_train=true`; val metrics are **leaky**). Effective batch **64** = `bs=16` × `grad_accum_steps=4`. Weights from `e4_ft.pt` (best 150-ep EMA).

```bash
cd "$REPO_ROOT"
LOG="logs/truite_e4_trainval_ft_${BATCH_TAG}.log"
PID="logs/truite_e4_trainval_ft_${BATCH_TAG}.pid"

nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$PY" -u -m smth2smth.pipelines.train \
  experiment=track_a_e4_tsm_trainval_ft track=a \
  > "$LOG" 2>&1 &
echo $! > "$PID"
```

Preset `track_a_e4_tsm_trainval_ft`: `lr=1e-5`, 25 ep, 3-ep warmup, `wd=0.01`. Ckpt: `e4_ft_trainval.pt`.

---

# Machine: thon — E5 (cRT: 3 phases)

## Machine prompt (paste on thon)

You are on VM **thon** for **E5**: (1) MAE pretrain 100 ep minimal aug → `e5_encoder.pt`; (2) champion FT **50 ep** instance-balanced → `e5_ft.pt`; (3) **cRT** 10 ep classifier-only, frozen backbone, `sqrt_inverse` sampler → `e5_ft_crt.pt`. Launch each phase with `nohup` only after the previous checkpoint exists. GPU host only.

**Logs:**

- `logs/thon_e5_mae_pretrain_${BATCH_TAG}.log`
- `logs/thon_e5_ft_stage1_rep50_${BATCH_TAG}.log`
- `logs/thon_e5_ft_stage2_crt10_${BATCH_TAG}.log`

### Phase 1 — MAE pretrain (~7 h)

```bash
cd "$REPO_ROOT"
LOG="logs/thon_e5_mae_pretrain_${BATCH_TAG}.log"
PID="logs/thon_e5_mae_pretrain_${BATCH_TAG}.pid"

nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$PY" -u -m smth2smth.pipelines.pretrain_videomae \
  experiment=track_a_ssl_pretrain_e5 track=a \
  > "$LOG" 2>&1 &
echo $! > "$PID"
```

### Phase 2a — FT stage 1 representation (~4 h)

**Wait for** `checkpoints/track_a/ssl/e5_encoder.pt`.

```bash
cd "$REPO_ROOT"
LOG="logs/thon_e5_ft_stage1_rep50_${BATCH_TAG}.log"
PID="logs/thon_e5_ft_stage1_rep50_${BATCH_TAG}.pid"

nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$PY" -u -m smth2smth.pipelines.train \
  experiment=track_a_ssl_finetune_e5 track=a \
  > "$LOG" 2>&1 &
echo $! > "$PID"
```

### Phase 2b — cRT stage 2 classifier (~0.7 h)

**Wait for** `checkpoints/track_a/ssl/e5_ft.pt` (stage-1 best).  
Stage 2 **resumes** that file, freezes encoder + attentive pool, writes **`e5_ft_crt.pt`**.

```bash
cd "$REPO_ROOT"
LOG="logs/thon_e5_ft_stage2_crt10_${BATCH_TAG}.log"
PID="logs/thon_e5_ft_stage2_crt10_${BATCH_TAG}.pid"

nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$PY" -u -m smth2smth.pipelines.train \
  experiment=track_a_ssl_finetune_e5_crt track=a \
  > "$LOG" 2>&1 &
echo $! > "$PID"
```

**Post-batch (thon only):** on honest val, sweep `tta_logit_adjust ∈ {0.0, 0.5, 1.0}` at submit time; pick τ with ≥0.2 pp gain else τ=0.

---

# Machine: Roussette — E6 (FT @256)

## Machine prompt (paste on Roussette)

You are on VM **Roussette** for **E6**: MAE pretrain @224 (100 ep, same as E1), then FT @**256** with `interpolate_pos_embed=true`, lighter reg (`drop_path=0.1`, `dropout=0.0`), 50 FT epochs. `nohup` on GPU host.

**Checkpoints:** `e6_encoder.pt`, `e6_ft.pt`  
**Logs:** `logs/roussette_e6_mae_pretrain_224_${BATCH_TAG}.log`, `logs/roussette_e6_ft_champion_256_${BATCH_TAG}.log`

### Phase 1 — MAE pretrain @224 (~7 h)

```bash
cd "$REPO_ROOT"
LOG="logs/roussette_e6_mae_pretrain_224_${BATCH_TAG}.log"
PID="logs/roussette_e6_mae_pretrain_224_${BATCH_TAG}.pid"

nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$PY" -u -m smth2smth.pipelines.pretrain_videomae \
  experiment=track_a_ssl_pretrain_e6 track=a \
  > "$LOG" 2>&1 &
echo $! > "$PID"
```

### Phase 2 — FT @256 (~5 h)

```bash
cd "$REPO_ROOT"
LOG="logs/roussette_e6_ft_champion_256_${BATCH_TAG}.log"
PID="logs/roussette_e6_ft_champion_256_${BATCH_TAG}.pid"

nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$PY" -u -m smth2smth.pipelines.train \
  experiment=track_a_ssl_finetune_e6 track=a \
  > "$LOG" 2>&1 &
echo $! > "$PID"
```

**Submit:** ViT TTA at base 256: `tta_scales=[0.875, 1.0, 1.125]` (224/256/288 — all ÷16). Confirm `[tta] ViT scale ...` lines in submit log.

---

# Machine: Raie — E7 (cosine mask schedule)

## Machine prompt (paste on Raie)

You are on VM **Raie** for **E7**: MAE pretrain **150 ep** with cosine mask ratio **0.90 → 0.75**, minimal aug, then champion FT 60 ep. **Start early** (~13–14 h). `nohup` on GPU host only.

**Checkpoints:** `e7_encoder.pt`, `e7_ft.pt`  
**Logs:** `logs/raie_e7_mae_pretrain_masksched_${BATCH_TAG}.log`, `logs/raie_e7_ft_champion_${BATCH_TAG}.log`

### Phase 1 — MAE pretrain with mask schedule (~8–9 h)

```bash
cd "$REPO_ROOT"
LOG="logs/raie_e7_mae_pretrain_masksched_${BATCH_TAG}.log"
PID="logs/raie_e7_mae_pretrain_masksched_${BATCH_TAG}.pid"

nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$PY" -u -m smth2smth.pipelines.pretrain_videomae \
  experiment=track_a_ssl_pretrain_e7 track=a \
  > "$LOG" 2>&1 &
echo $! > "$PID"
```

**Monitor for:** `[videomae] epoch k: mask_ratio=...` on epochs 1, 150, and every 50.

### Phase 2 — Champion FT (~4.5 h)

```bash
cd "$REPO_ROOT"
LOG="logs/raie_e7_ft_champion_${BATCH_TAG}.log"
PID="logs/raie_e7_ft_champion_${BATCH_TAG}.pid"

nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$PY" -u -m smth2smth.pipelines.train \
  experiment=track_a_ssl_finetune_e7 track=a \
  > "$LOG" 2>&1 &
echo $! > "$PID"
```

---

## Optional: chained launcher (E1 example — Anchois)

If you prefer **one** background shell that runs pretrain then FT (still `nohup`, still unbuffered):

```bash
cd "$REPO_ROOT"
LOG="logs/anchois_e1_chain_${BATCH_TAG}.log"
PID="logs/anchois_e1_chain_${BATCH_TAG}.pid"

nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src bash -c '
  set -euo pipefail
  PY="'"$PY"'"
  echo "[chain] $(date -Is) Phase 1 MAE pretrain"
  "$PY" -u -m smth2smth.pipelines.pretrain_videomae \
    experiment=track_a_ssl_pretrain_e1 track=a
  test -s checkpoints/track_a/ssl/e1_encoder.pt
  echo "[chain] $(date -Is) Phase 2 champion FT"
  "$PY" -u -m smth2smth.pipelines.train \
    experiment=track_a_ssl_finetune_e1 track=a
  echo "[chain] $(date -Is) Done."
' > "$LOG" 2>&1 &
echo $! > "$PID"
```

Adapt `experiment=` / paths for other machines (E5 needs **three** sequential steps — do **not** chain all three unless you add waits and distinct log sections).

---

## Quick reference — Hydra experiment names

| Phase        | Hydra `experiment=`                      |
|-------------|-------------------------------------------|
| E1 pretrain | `track_a_ssl_pretrain_e1`             |
| E1 FT       | `track_a_ssl_finetune_e1`             |
| E2 pretrain | `track_a_ssl_pretrain_e2`            |
| E2 FT       | `track_a_ssl_finetune_e2`            |
| E3 pretrain | `track_a_ssl_pretrain_e3`             |
| E3 FT       | `track_a_ssl_finetune_e3`             |
| E4 train    | `track_a_e4_tsm_sgdr`                        |
| E4 cont.    | `track_a_e4_tsm_continue150`                 |
| E4 trainval | `track_a_e4_tsm_trainval_ft`                 |
| E5 pretrain | `track_a_ssl_pretrain_e5`            |
| E5 FT s1    | `track_a_ssl_finetune_e5`            |
| E5 FT s2    | `track_a_ssl_finetune_e5_crt`        |
| E6 pretrain | `track_a_ssl_pretrain_e6`             |
| E6 FT       | `track_a_ssl_finetune_e6`             |
| E7 pretrain | `track_a_ssl_pretrain_e7`           |
| E7 FT       | `track_a_ssl_finetune_e7`           |

Always append: **`track=a`**

---

## If `uv run` is available on PATH

Replace `"$PY" -u -m` with:

```bash
uv run python -u -m
```

keeping `env PYTHONUNBUFFERED=1 PYTHONPATH=src` in the `nohup` line.

---

## Do not

- Run these commands inside Cursor’s sandbox expecting GPU training.
- Point any `init_from` / `resume_from` at another machine’s checkpoints.
- Use `dataset.include_val_in_train=true` or train on val labels.
- Use ViT TTA `[0.875, 1.0, 1.125]` at base resolution 224 (PatchEmbed crash risk).
- Relaunch the live Phase-2 R50+TSM 150-ep job (not part of this batch).
