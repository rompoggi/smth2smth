# Track B — Round-1 overnight grid (12 independent runs)

**Spec:** [`track_b_round1.md`](track_b_round1.md) · **W&B:** project `smth2smth-track-b`, group `track_b_r1_overnight`  
**Logs:** `logs/track_b/r1_<RUN>_20260527.log`

---

## Frozen holdout (`holdout_clean`)

All round-1 Track B runs use the **same** validation holdout (Exp 8 in `track_b_round1.md`):

| Item | Value |
|------|--------|
| File | `data/holdout_clean.json` (commit to git) |
| Split | **15%** of `data/val`, **stratified per class**, **seed 42** |
| Holdout size | **1013** / 6745 val clips |
| Train pool | `data/train` (44993) + **5732** val clips not in holdout |

**Distribution check** (see `outputs/holdout_clean/distribution_report.json`):

- Max per-class holdout fraction error vs 15%: **0.71 pp**
- Max \|p_train − p_val\| per class: **2.29 pp** (32 classes present in both splits)

Rebuild only if `data/val` changes:

```bash
PYTHONPATH=src uv run python scripts/build_holdout_clean.py
```

Hydra: all `track_b_r1_*` experiments compose `track_b_r1_data` → `holdout_clean` data config.  
`train.py` uses `holdout_manifest` (frozen keys); **ignores** per-run `seed` for the holdout split.

---

## Prerequisite: commit + push from gymnote

These paths must be on `origin/main` before `git pull` on fish hosts:

- `data/holdout_clean.json`
- `configs/data/holdout_clean.yaml`, `configs/experiment/track_b_r1_*.yaml`
- `src/smth2smth/ensemble/holdout.py`, `src/smth2smth/pipelines/train.py`
- `src/smth2smth/track_b/vjepa2.py` (`vjepa2_hf_clf` model alias)
- `scripts/build_holdout_clean.py`, `scripts/verify_track_b_r1_host.sh`, `scripts/launch_track_b_r1_fleet.sh`
- `tests/ensemble/test_holdout_clean.py`
- `RUN_track_b_r1_overnight.md`

Then on gymnote:

```bash
cd /Data/romain.poggi/smth2smth
git pull   # sync your push
bash scripts/launch_track_b_r1_fleet.sh verify   # all 12 must print OK
bash scripts/launch_track_b_r1_fleet.sh launch
```

---

## Host assignment

| Host | Run | `experiment=` | Varies |
|------|-----|---------------|--------|
| **anguille** | `r1_b01_hf_lr5e3` | `track_b_r1_b01_hf_lr5e3` | LR 5e-3 / LoRA 5e-4 |
| **barbeau** | `r1_b02_hf_lr3e3` | `track_b_r1_b02_hf_lr3e3` | LR 3e-3 / LoRA 3e-4 |
| **barbue** | `r1_b03_hf_lr1e3` | `track_b_r1_b03_hf_lr1e3` | LR 1e-3 / LoRA 1e-4 |
| **baudroie** | `r1_b04_hf_lr3e4` | `track_b_r1_b04_hf_lr3e4` | LR 3e-4 / LoRA 3e-5 |
| **gardon** | `r1_b05_hf_lr1e4` | `track_b_r1_b05_hf_lr1e4` | LR 1e-4 / LoRA 1e-5 |
| **lieu** | `r1_b06_hf_wd001` | `track_b_r1_b06_hf_wd001` | WD 0.01 |
| **murene** | `r1_b07_mq16_lr1e3` | `track_b_r1_b07_mq16_lr1e3` | MQ Q=16 |
| **piranha** | `r1_b08_mq16_lr3e3` | `track_b_r1_b08_mq16_lr3e3` | MQ Q=16, LR 3e-3 |
| **raie** | `r1_b09_mq8_lr1e3` | `track_b_r1_b09_mq8_lr1e3` | MQ Q=8 |
| **roussette** | `r1_b10_mq32_lr1e3` | `track_b_r1_b10_mq32_lr1e3` | MQ Q=32 |
| **saumon** | `r1_b11_4block_lr3e4` | `track_b_r1_b11_4block_lr3e4` | 4-block probe |
| **truite** | `r1_b12_hf_dora_mlp` | `track_b_r1_b12_hf_dora_mlp` | DoRA + MLP LoRA |

---

## Preflight checklist (per host)

| Check | Command / criterion |
|-------|---------------------|
| Data dir perms | `stat -c '%a' /Data/romain.poggi` → **700** |
| Repo | `/Data/romain.poggi/smth2smth/.git` |
| W&B | `.env` contains `WANDB_API_KEY=...` |
| Data | `data/train` has **≥32** class folders (professor 33-class set; **no** full `data/ssv2` required) |
| Venv | `.venv/bin/python` or `uv sync` |
| Configs | `configs/experiment/track_b_r1_b01_hf_lr5e3.yaml` after `git pull` |

**Data source for copy:** use **anguille** (or any fish with `train_classes=32`, no `data/ssv2`). **Do not** rsync `data/` from **gymnote** (includes full SSv2 under `data/ssv2/`).

```bash
# Example: gardon missing data (from gardon)
rsync -az anguille:/Data/romain.poggi/smth2smth/data/ /Data/romain.poggi/smth2smth/data/
```

---

## Setup status (2026-05-27 audit)

| Host | perm | git | venv | wandb | train (classes) | Notes |
|------|------|-----|------|-------|-----------------|-------|
| anguille | 700 | yes | yes | yes | 32 | OK — data source |
| barbeau | 700 | yes | yes | yes | 32 | OK |
| barbue | 700 | yes | yes | yes | 32 | OK |
| baudroie | 700 | yes | yes | yes | 32 | OK |
| gardon | 700 | yes | yes | yes | rsync | `data/` copying from anguille |
| lieu | 700 | yes | yes | yes | 32 | OK |
| murene | 700 | yes | yes | yes | 32 | OK |
| piranha | 700 | yes | yes | yes | 32 | OK |
| raie | 700 | yes | yes | yes | 32 | OK |
| roussette | 700 | yes | yes | yes | 32 | OK |
| saumon | 700 | yes | yes | yes | 32 | OK |
| truite | 700 | yes | yes | yes | 32 | OK |

**Blocked until push:** `track_b_r1_*` YAMLs not on `origin/main` yet.

---

## Manual single-host launch

```bash
DATA=/Data/romain.poggi
REPO=/Data/romain.poggi/smth2smth
RUN=r1_b01_hf_lr5e3
EXP=track_b_r1_b01_hf_lr5e3
TAG=20260527
cd "$REPO"
chmod 700 "$DATA"
git pull
bash scripts/verify_track_b_r1_host.sh
LOG=logs/track_b/${RUN}_${TAG}.log
mkdir -p logs/track_b checkpoints/track_b
{
  echo "# run: $RUN"
  echo "# started: $(date -Is)"
  echo "# track: b"
  echo "# experiment_doc: track_b_round1.md"
  echo "# hydra: experiment=$EXP"
} >"$LOG"
nohup env PYTHONPATH=src PYTHONUNBUFFERED=1 .venv/bin/python -u -m smth2smth.pipelines.train \
  track=b experiment=$EXP training.checkpoint_path=checkpoints/track_b/${RUN}.pt \
  >>"$LOG" 2>&1 &
```

---

## Monitor

```bash
ssh anguille 'tail -30 /Data/romain.poggi/smth2smth/logs/track_b/r1_b01_hf_lr5e3_20260527.log'
# W&B: group track_b_r1_overnight
```
