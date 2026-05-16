# Track A — VideoMAE SSL night sweep (16 VMs)

This document is for a **fresh VM** with the repo cloned: it explains **what** you are running, **which Hydra presets** apply to your machine, and **how** to launch an unbuffered two-phase job (MAE pretrain → supervised fine-tune with **attentive probe**).

---

## 1. Experiment context (30 seconds)

- **Track:** A (closed world — no external pretraining).
- **Phase A — SSL:** `smth2smth.pipelines.pretrain_videomae` trains a **VideoMAE** encoder on **all** clips (train + val + test folders), masked reconstruction.
- **Phase B — Supervised FT:** `smth2smth.pipelines.train` loads that encoder via `model.init_from`, uses **`model.head=attn`** (attentive probe) with **`head_num_heads`** set per slot (see table).
- **Independence:** Each codename is a **full pipeline** on **one** machine. No VM reads another VM’s checkpoints.
- **Artifacts:** Encoder and fine-tune weights are written under  
  `checkpoints/track_a/ssl/<fish>_encoder.pt` and `checkpoints/track_a/ssl/<fish>_ft.pt`  
  (see generated `configs/experiment/track_a_ssl_*_<fish>.yaml`).

Dataset paths and environment setup follow the project **README** (this file does not duplicate cluster-specific directories).

---

## 2. Which run is mine?

You were assigned a **codename** (fish) and a **person** (Thomas or Romain). Use the table once to map **codename → Hydra experiment names**.

### Thomas (slots 1–8)

| Slot | Codename | Phase 1 `experiment=` | Phase 2 `experiment=` |
|------|----------|------------------------|------------------------|
| 1 | **saumon** | `track_a_ssl_pretrain_saumon` | `track_a_ssl_finetune_saumon` |
| 2 | **silure** | `track_a_ssl_pretrain_silure` | `track_a_ssl_finetune_silure` |
| 3 | **gymnote** | `track_a_ssl_pretrain_gymnote` | `track_a_ssl_finetune_gymnote` |
| 4 | **doubs** | `track_a_ssl_pretrain_doubs` | `track_a_ssl_finetune_doubs` |
| 5 | **ablette** | `track_a_ssl_pretrain_ablette` | `track_a_ssl_finetune_ablette` |
| 6 | **anchois** | `track_a_ssl_pretrain_anchois` | `track_a_ssl_finetune_anchois` |
| 7 | **anguille** | `track_a_ssl_pretrain_anguille` | `track_a_ssl_finetune_anguille` |
| 8 | **barbeau** | `track_a_ssl_pretrain_barbeau` | `track_a_ssl_finetune_barbeau` |

### Romain (slots 9–16)

| Slot | Codename | Phase 1 `experiment=` | Phase 2 `experiment=` |
|------|----------|------------------------|------------------------|
| 9 | **truite** | `track_a_ssl_pretrain_truite` | `track_a_ssl_finetune_truite` |
| 10 | **roussette** | `track_a_ssl_pretrain_roussette` | `track_a_ssl_finetune_roussette` |
| 11 | **rouget** | `track_a_ssl_pretrain_rouget` | `track_a_ssl_finetune_rouget` |
| 12 | **raie** | `track_a_ssl_pretrain_raie` | `track_a_ssl_finetune_raie` |
| 13 | **sole** | `track_a_ssl_pretrain_sole` | `track_a_ssl_finetune_sole` |
| 14 | **thon** | `track_a_ssl_pretrain_thon` | `track_a_ssl_finetune_thon` |
| 15 | **piranha** | `track_a_ssl_pretrain_piranha` | `track_a_ssl_finetune_piranha` |
| 16 | **murene** | `track_a_ssl_pretrain_murene` | `track_a_ssl_finetune_murene` |

**Quick lookup from the shell** (replace the codename if needed):

```bash
cd /path/to/smth2smth
ls configs/experiment/track_a_ssl_pretrain_<CODENAME>.yaml configs/experiment/track_a_ssl_finetune_<CODENAME>.yaml
```

If both files exist, you have the right pair. Open them to see **ViT-S vs ViT-B**, **seed**, **MAE schedule**, and **probe head width** (`head_num_heads`).

---

## 3. Where the configs live

- **Directory:** `configs/experiment/`
- **Naming:**  
  - `track_a_ssl_pretrain_<codename>.yaml` — VideoMAE only.  
  - `track_a_ssl_finetune_<codename>.yaml` — attentive fine-tune; `model.init_from` points at the encoder path produced by the matching pretrain file.

Do **not** point Phase 2 at another codename’s encoder path.

---

## 4. One-shot shell: create dirs, Phase 1, then Phase 2

From the **repository root**, with dependencies installed per README (`uv sync`, `.venv`, dataset env vars/paths configured).

```bash
export PYTHONUNBUFFERED=1
export CODENAME=truite          # <-- set to YOUR codename
mkdir -p logs checkpoints/track_a/ssl

# Phase 1 — VideoMAE SSL (log file should receive lines immediately)
nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src uv run python -u -m smth2smth.pipelines.pretrain_videomae \
  experiment=track_a_ssl_pretrain_${CODENAME} track=a \
  > logs/ssl_${CODENAME}_pretrain.log 2>&1 &
echo $! > logs/ssl_${CODENAME}_pretrain.pid
```

Wait until pretrain finishes and **`checkpoints/track_a/ssl/${CODENAME}_encoder.pt`** exists (and the log shows the final `wrote encoder checkpoint` line).

```bash
# Phase 2 — Attentive probe fine-tune
nohup env PYTHONUNBUFFERED=1 PYTHONPATH=src uv run python -u -m smth2smth.pipelines.train \
  experiment=track_a_ssl_finetune_${CODENAME} track=a \
  > logs/ssl_${CODENAME}_finetune.log 2>&1 &
echo $! > logs/ssl_${CODENAME}_finetune.pid
```

**Why two layers of unbuffered output**

- **`PYTHONUNBUFFERED=1`** — Python does not block-write stdout in chunks.
- **`-u`** on the interpreter — unbuffered stdin/stdout/stderr (redundant with `PYTHONUNBUFFERED` but makes intent obvious in `ps`).
- **`nohup ... > logs/... 2>&1`** — persist everything; `tail -f` works on a file, not a tty.

---

## 5. Verify the process actually started

Within a few seconds:

```bash
# Replace with your codename and phase log
tail -n 40 logs/ssl_truite_pretrain.log
```

You should see **Hydra composed YAML** printed first, then lines like `[videomae] scanning video folders` and dataloader activity.

```bash
pgrep -af "smth2smth.pipelines.pretrain_videomae|smth2smth.pipelines.train"
```

```bash
# Optional: GPU memory in use
nvidia-smi
```

```bash
kill -0 "$(cat logs/ssl_truite_pretrain.pid)" && echo still_running
```

---

## 6. Timestamps every 50 epochs (and first / last)

The trainers write **ISO-8601 timestamps** on epoch progress lines on a **50-epoch grid**, and **always** on **epoch 1** and the **last** epoch of each phase:

- **Pretrain** (`pretrain_videomae`): lines like  
  `[videomae] 2026-05-16T23:01:02 epoch 50/100 avg loss ...`
- **Fine-tune** (`train`): lines like  
  `[2026-05-16T23:15:00] Epoch 50/30 | train loss ...`  
  (for standard 30-epoch FT, expect a timestamp on **epoch 1** and **epoch 30**; intermediate 50s appear only if `training.epochs` ≥ 50).

Between milestone epochs, pretrain still prints a short **non-timestamped** `avg loss` line each epoch so you can detect stalls without flooding logs.

Step-level logs (`log_interval_steps`) do not add timestamps on every step; rely on epoch summaries for wall-clock checkpoints.

---

## 7. Slot cheat-sheet (what differs per codename)

Summarized; full detail is in each YAML.

- **Thomas (1–8):** ViT-B slots; **silure** uses `seed=123`; **gymnote** Long MAE (`pretrain.epochs=200`); **doubs** / **ablette** wider attentive head (8 / 12 heads); **anchois** MAE `mask_ratio=0.85`; **anguille** MAE `lr=1e-4`; **barbeau** MAE `warmup_epochs=10`.
- **Romain (9–16):** ViT-S slots; **roussette** `seed=123`; **rouget** long MAE; **raie** / **sole** probe heads 6 / 8; **thon** MAE masking; **piranha** FT `epochs=40`; **murene** FT `lr=3e-4`.

---

## 8. If something fails

- **Empty log / no first lines:** check you used **`>` not mistaken path, `cd` to repo root, `PYTHONPATH=src`.
- **CUDA OOM:** lower `pretrain.batch_size` or `training.batch_size` via Hydra CLI override (same codename).
- **Missing videos:** pretrain exits with `No video folders found` — fix `dataset.*` paths in root config or overrides (see README).
