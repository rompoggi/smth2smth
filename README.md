# smth2smth — Something-Something V2 action anticipation

Code for the **CSC_43M04_EP — Modal d'informatique — Deep Learning in Computer Vision**
challenge *"Can a model predict the future? / What Happens Next?"*, run as a
Kaggle competition on a curated 50-class (33 used on disk) subset of the
**Something-Something V2** dataset. Given only the first frames of a clip, the
model must predict the action that happens next (action **anticipation**, not
recognition). Primary metric: **Top-1 accuracy**; secondary: Top-5.

The whole project is a single Python package (`smth2smth`) driven by
[Hydra](https://hydra.cc) configs. One generic training/eval/submit pipeline
serves **both competition tracks**; everything track- or experiment-specific
lives in YAML under `configs/`.

**Authors:** Romain Poggi, Thomas Turkieh

> **Reading guide for the grader.** This README is the operational manual: how
> to set up, train, evaluate, submit, and reproduce every experiment. The
> *scientific* narrative (motivation, equations, ablations, per-run logbook) is
> in `report/` (LaTeX → compile `report/main.tex`). Every committed prediction
> file is in `submissions/`.

---

## Table of contents

1. [The two tracks](#the-two-tracks)
2. [Environment setup](#1-environment-setup)
3. [Data](#2-data)
4. [Repository layout](#3-repository-layout)
5. [The pipelines (entry points)](#4-the-pipelines-entry-points)
6. [Quick start — baselines](#5-quick-start--baselines)
7. [Models & experiment catalog](#6-models--experiment-catalog)
8. [Reproducing each kind of experiment](#7-reproducing-each-kind-of-experiment)
9. [Configuration reference (Hydra knobs)](#8-configuration-reference-hydra-knobs)
10. [Submissions & validation](#9-submissions--validation)
11. [Results](#10-results)
12. [Testing, linting, troubleshooting](#11-testing-linting-troubleshooting)

---

## The two tracks

| | **Track A — Closed World** | **Track B — Open World** |
| --- | --- | --- |
| Pretrained weights | **not allowed** | allowed |
| External data | **not allowed** | allowed |
| Headline approach | from-scratch TSM-ResNet50 / VideoMAE SSL pretrain + finetune | frozen **V-JEPA 2** encoder + probe / LoRA |
| Checkpoints | `checkpoints/track_a/` | `checkpoints/track_b/` |
| Submissions | `submissions/track_a.csv` | `submissions/track_b.csv` |

Tracks are **output-scoping overlays** (`track=a` / `track=b` set checkpoint and
submission paths and Track-A's default TTA). The *from-scratch vs pretrained*
choice is encoded in the **experiment** file, so you can legally run any
experiment under either track for ablation.

---

## 1. Environment setup

The project pins **Python 3.12.9** and uses [`uv`](https://docs.astral.sh/uv/)
for dependency management (lockfile: `uv.lock`).

```bash
# from the repo root
uv venv --python 3.12.9
uv sync                       # installs the exact locked dependency set
```

GPU notes:

- `pyproject.toml` pulls **PyTorch ≥ 2.9 built for CUDA 12.8** (`cu128` wheels)
  on Linux/Windows; macOS falls back to CPU wheels automatically.
- A CUDA GPU is required to *train* the large models in a reasonable time. All
  pipelines fall back to CPU if CUDA is missing (`training.device=cpu`), which
  is enough for the unit tests and tiny smoke runs.
- **Track B** downloads V-JEPA 2 weights from the HuggingFace Hub on first use
  (≈1.2 GiB for ViT-L, larger for ViT-g) and caches them under `$HF_HOME`.

Optional **Weights & Biases** logging: copy `.env.example` → `.env` and fill in
your keys. Logging is off unless `WANDB_API_KEY` is set; nothing in the core
pipelines requires it.

The package is intentionally **not installed** (`[tool.uv] package = false`), so
every command runs with `PYTHONPATH=src`. The `scripts/run_track_*.py` wrappers
set this for you.

---

## 2. Data

Place the frame-extracted dataset under `data/` (gitignored). Expected layout:

```text
data/
  train/<NNN_ClassName>/<video_id>/frame_*.jpg   # class-bucketed
  val/<NNN_ClassName>/<video_id>/frame_*.jpg     # official held-out val
  test/<video_id>/frame_*.jpg                    # flat, no labels
```

If your data already lives elsewhere, **symlink** instead of copying:

```bash
ln -s /absolute/path/to/frames data
```

A helper to fetch the course dataset from Google Drive (kept for reference, may
need a fresh file id):

```bash
uv run scripts/download_data.py        # downloads + unzips into ./data
```

**Dataset facts worth knowing (they explain several config choices):**

- **33 class indices** are present on disk (the "50 classes" of the brief), and
  **class 27 has no training samples**. The trainer records the trained class
  set in the checkpoint and the submission pipeline masks never-trained classes
  to `-inf` before argmax, so the model can never accidentally predict class 27.
- Each clip stores only **~4 real frames** on disk; the dataloader expands them
  to the requested `dataset.num_frames` via linspace index sampling. Direction-
  only classes (opening vs closing, etc.) are information-limited at the source
  — this motivates the temporal/time-reversal augmentations.
- **Official validation:** by default the trainer carves an internal 80/20 split
  out of `train/`. Set `dataset.use_official_val=true` to validate on the real
  `val/` folder instead — this is what makes the validation score track the
  Kaggle leaderboard (see `report/track_a.tex`, "Official validation split").

---

## 3. Repository layout

```text
smth2smth/
├── configs/                      # Hydra YAML (the entire experiment surface)
│   ├── config.yaml               # root composition + defaults
│   ├── data/                     # dataset paths, splits, augmentation hooks
│   ├── train/                    # optimizer / schedule / AMP / EMA / TTA knobs
│   ├── model/                    # one file per architecture
│   ├── augment/                  # spatial/temporal augmentation presets
│   ├── pretrain/                 # SSL pretraining (VideoMAE / V-JEPA) configs
│   ├── test/                     # test-time augmentation presets
│   ├── track/{a,b}.yaml          # output-path overlays
│   └── experiment/               # ~45 named, reproducible experiment bundles
├── src/smth2smth/
│   ├── pipelines/                # Hydra entry points (the only Hydra-aware layer)
│   │   ├── train.py              # supervised training (all models)
│   │   ├── evaluate.py           # Top-1/Top-5 on val from a checkpoint
│   │   ├── submit.py             # test inference → submission CSV (+ TTA)
│   │   ├── pretrain_videomae.py  # VideoMAE masked-reconstruction SSL (Track A)
│   │   ├── pretrain_vjepa.py     # V-JEPA-style clip SSL (Track A)
│   │   └── pretrain_ssl.py       # legacy DINO-on-frames SSL (deprecated)
│   ├── shared/
│   │   ├── data/                 # datasets, transforms, augmentations
│   │   ├── models/               # architectures + registry (build_model)
│   │   ├── engine/               # train/eval loops, metrics, stability logging
│   │   ├── io/                   # checkpoint + submission read/write/validate
│   │   └── utils/                # seeding, splits, class balance, W&B
│   ├── track_a/                  # Track-A-specific code (currently overlays)
│   ├── track_b/                  # V-JEPA 2 model + zero-shot helpers
│   ├── ensemble/                 # multi-model logit caching / weighting / submit
│   └── analysis/                 # head parameter analysis
├── scripts/                      # track wrappers + dev tools
├── tests/                        # pytest suite mirroring src/ (284 tests)
├── report/                       # LaTeX report + figures (the science)
├── submissions/                  # every Kaggle submission CSV produced
├── docs/                         # architecture notes, challenge brief, diagrams
├── pyproject.toml                # deps, pytest, ruff config
└── uv.lock
```

See [`docs/architecture.md`](docs/architecture.md) for the module dependency
graph and data flow. Architectural rule: **only `pipelines/` knows about
Hydra**; `shared/` takes plain Python objects and is fully unit-tested.

---

## 4. The pipelines (entry points)

Every pipeline is a Hydra app reading the same composed config. Generic form:

```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.<name> [hydra overrides...]
```

| Module | Purpose | Reads | Writes |
| --- | --- | --- | --- |
| `pipelines.train` | supervised training (any registered model) | `data/`, `model`, `train`, `augment`, `experiment`, `track` | `checkpoints/track_*/...pt` |
| `pipelines.evaluate` | Top-1/Top-5 on the val split from a checkpoint | a checkpoint | stdout report |
| `pipelines.submit` | inference on `test/` → submission CSV (with TTA) | a checkpoint, `test` group | `submissions/track_*.csv` |
| `pipelines.pretrain_videomae` | VideoMAE masked SSL (encoder-only ckpt) | `pretrain=videomae` | SSL encoder `.pt` |
| `pipelines.pretrain_vjepa` | V-JEPA-style clip SSL (trunk-only ckpt) | `pretrain=vjepa` | SSL trunk `.pt` |
| `pipelines.pretrain_ssl` | legacy DINO-on-frames SSL (deprecated) | `pretrain=default` | SSL trunk `.pt` |

SSL checkpoints feed back into supervised training via `model.init_from=<path>`
(non-strict load; the trainer renames keys as needed). This is the Track-A
"pretrain on the provided unlabeled frames, then finetune" workflow.

**Inspect the fully-resolved config without running anything** (do this first
whenever you adapt a command — it shows exactly what will run):

```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.train \
    --cfg job --resolve experiment=track_b_vjepa2_ssv2ft_lora16f track=b
```

---

## 5. Quick start — baselines

The two baselines defined by the competition. The wrapper scripts pin the right
`track=` + `experiment=` and set `PYTHONPATH`; append any Hydra override.

```bash
# Track A — from scratch (closed world)
python scripts/run_track_a.py training.epochs=20
# Track B — ImageNet-pretrained ResNet baseline (open world)
python scripts/run_track_b.py training.epochs=20
```

Then evaluate and produce a submission (Track A shown; swap `a`→`b`):

```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.evaluate \
    track=a training.checkpoint_path=checkpoints/track_a/best_model.pt
PYTHONPATH=src uv run python -m smth2smth.pipelines.submit \
    track=a training.checkpoint_path=checkpoints/track_a/best_model.pt
# -> submissions/track_a.csv
```

`evaluate` prints `Validation samples`, `Top-1 accuracy`, `Top-5 accuracy`,
`Loss`. `submit` writes the CSV and re-validates it before exiting.

Sanity-check the model registry at any time:

```bash
PYTHONPATH=src uv run python -c \
  "from smth2smth.shared.models import list_registered_models; print(list_registered_models())"
# ['avanced_resnet50_tsm', 'cnn_baseline', 'cnn_lstm', 'dual_stream_rgb_diff_tsm',
#  'video_mae_vit', 'vjepa2', 'vjepa2_hf_clf', 'vjepa2_ssv2ft']
```

---

## 6. Models & experiment catalog

### Registered models (`configs/model/*.yaml`, selected with `model=<name>`)

| `model.name` | Description | Track |
| --- | --- | --- |
| `cnn_baseline` | ResNet-18 frame encoder + temporal average pool | A/B |
| `cnn_lstm` | ResNet-18 frame encoder + LSTM over frame embeddings | A/B |
| `avanced_resnet50_tsm` | ResNet-50 + Temporal Shift Modules, attn/mean head, DropPath, EMA | A |
| `dual_stream_rgb_diff_tsm` | RGB ResNet-50 + frame-difference ResNet-34, both TSM, fused | A |
| `video_mae_vit` | ViT (VideoMAE) encoder; supports SSL warm-start + diverse heads (Perceiver/divided-ST/AIM) and LLRD finetune | A |
| `vjepa2` / `vjepa2_ssv2ft` | frozen V-JEPA 2 encoder (HF) + attentive/linear probe, optional LoRA/DoRA | B |
| `vjepa2_hf_clf` | V-JEPA 2 with the HF `VJEPA2ForVideoClassification` head | B |

### Experiment bundles (`configs/experiment/*.yaml`, selected with `experiment=<name>`)

Each file composes a model + augment + train recipe into one reproducible name.
List them all:

```bash
ls configs/experiment/
```

Highlights (the rest follow the same `track_a_*` / `track_b_*` naming):

**Track A**
- `baseline_from_scratch` — the closed-world baseline.
- `track_a_phase2` / `track_a_phase2_balanced` — TSM-ResNet50 + AMP + EMA + attn head + DropPath + AdamW.
- `track_a_videomae_pretrain` / `track_a_videomae_pretrain_resnet` — VideoMAE SSL pretraining.
- `track_a_videomae_finetune` / `track_a_videomae_official_ssv2_ft` / `track_a_mae25_ft` — finetune from a VideoMAE encoder.
- `track_a_videomae_submit_champion` — submission recipe for the champion VideoMAE run.
- `track_a_vjepa_pretrain` / `track_a_vjepa_finetune` — V-JEPA-style SSL + finetune (Phase 4 augment bundle).
- `track_a_ssl_pretrain` / `track_a_ssl_finetune` — legacy DINO SSL path.
- `track_a_dual_stream_30e_class_boost` / `..._long_class_boost` — dual-stream RGB+diff with class boosting.
- `track_a_diverse_arch{1..4}_*` — diverse heads/temporal modules for ensembling (attn-probe, Perceiver, divided space-time, AIM).
- `track_a_hc_ablation_{baseline,shc,mhc}` — Hyper-Connection routing ablations.

**Track B**
- `baseline_pretrained` — ImageNet ResNet baseline.
- `track_b_vjepa2` — frozen V-JEPA 2 ViT-L attentive probe (the canonical Track-B run).
- `track_b_vjepa2_heavy_aug` / `track_b_vjepa2_vitg384_heavy_aug` — heavy augmentation + CubeCutMix, ViT-L / ViT-g 384.
- `track_b_vjepa2_ssv2ft_lora16f` / `track_b_vjepa2_hfclf_{8,16}f_lora_r16` — SSv2-finetuned encoder + LoRA r16.
- `track_b_vjepa2_vitl_30e_warmup` / `..._fulltrain` / `..._fulltrain_lowlr` / `..._ema9998` — ViT-L training variants.
- `track_b_vjepa2_peft` / `track_b_vjepa2_lora16_mlp` / `track_b_vjepa2_dora16_mlp` — multi-query probe + PEFT LoRA/DoRA.
- `track_b_vjepa2_4block_lastk2` / `track_b_vjepa2_ssv2ft_frozen_probe` / `track_b_vjepa2_ssv2ft_extra_train` — probe-depth / frozen-probe / external-data (E6) variants.

---

## 7. Reproducing each kind of experiment

> Every command below assumes the [environment](#1-environment-setup) and
> [data](#2-data) are in place. For long jobs, launch detached so the run
> survives a disconnect (pattern at the end of this section).

### 7.1 Supervised training (any model/experiment)

```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.train \
    track=a experiment=track_a_phase2_balanced
```

The trainer prints the resolved config, the optimizer param-group breakdown, and
per-epoch Top-1/Top-5; it saves the **best-by-val-Top-1** checkpoint (live or EMA,
whichever is higher) to `training.checkpoint_path`. Optimizer / scheduler / AMP
scaler state are checkpointed so runs resume cleanly.

### 7.2 Track A — self-supervised pretrain → finetune (closed world)

Track A forbids external data and pretrained weights, but **does** allow
self-supervised pretraining on the *provided* unlabeled frames (train+val+test).

```bash
# (a) VideoMAE masked-reconstruction pretraining → encoder-only checkpoint
PYTHONPATH=src uv run python -m smth2smth.pipelines.pretrain_videomae \
    experiment=track_a_videomae_pretrain

# (b) supervised finetune, warm-starting from the SSL encoder
PYTHONPATH=src uv run python -m smth2smth.pipelines.train \
    track=a experiment=track_a_videomae_finetune \
    model.init_from=checkpoints/track_a/<videomae_encoder>.pt
```

The V-JEPA-style trunk SSL is the same shape:
`pretrain_vjepa experiment=track_a_vjepa_pretrain`, then
`train experiment=track_a_vjepa_finetune model.init_from=<trunk>.pt`.
(`pretrain_ssl` / DINO is kept for the record but deprecated — see the report.)

### 7.3 Track B — V-JEPA 2 (open world)

```bash
# Canonical frozen-encoder attentive probe (ViT-L, downloads weights on first run)
PYTHONPATH=src uv run python -m smth2smth.pipelines.train \
    track=b experiment=track_b_vjepa2

# SSv2-finetuned encoder + LoRA r16 (a strong Track-B recipe)
PYTHONPATH=src uv run python -m smth2smth.pipelines.train \
    track=b experiment=track_b_vjepa2_ssv2ft_lora16f
```

Three V-JEPA 2 settings are **non-negotiable** (enforced by the presets):
`model.pretrained=true` (ImageNet normalization the encoder expects),
`dataset.image_size` ∈ {256, 384} matching the chosen `hf_repo`, and an **even**
`dataset.num_frames` (tubelet size 2). Swap the backbone with
`model.hf_repo=facebook/vjepa2-vitg-fpc64-384-ssv2 dataset.image_size=384`.

### 7.4 Zero-shot Track B (no training)

The SSv2-finetuned V-JEPA 2 checkpoint ships a 174-class head; our 33 classes are
a strict subset, so it can be submitted with no training. Helpers live in
`src/smth2smth/track_b/zero_shot.py`
(`build_label_mapping`, `VideoFramesDataset`); the committed result is
`submissions/track_b_vjepa2_vitl_fpc16_256_ssv2_zero_shot.csv` (~45% Kaggle).
See `report/track_b.tex` §"Zero-shot Evaluation".

### 7.5 Evaluate / submit (with TTA)

```bash
# Evaluate Top-1/Top-5 on the val split
PYTHONPATH=src uv run python -m smth2smth.pipelines.evaluate \
    track=b training.checkpoint_path=checkpoints/track_b/best_model.pt

# Submit. Track A enables multi-scale + flip TTA by default (track/a.yaml);
# add flip TTA explicitly elsewhere via training.tta / training.tta_flip.
PYTHONPATH=src uv run python -m smth2smth.pipelines.submit \
    track=b training.checkpoint_path=checkpoints/track_b/best_model.pt
```

TTA correctly handles direction-sensitive classes: the horizontal-flip view's
softmax is remapped through an auto-derived left↔right class permutation before
averaging, so flipping "pull left→right" votes for "pull right→left". TSN-style
dense TTA (multi-segment × multi-crop) is controlled by the `test` group
(`test=videomae_official_2x3` etc.).

> **Empirical note for this dataset: NoTTA often wins.** Several of our best
> submissions used no TTA — flip/scale averaging hurt on the motion-direction
> classes. Compare a `notta` and a `tta` submission before choosing (see
> `submissions/*_notta_*.csv`).

### 7.6 Ensembling

`src/smth2smth/ensemble/` provides the multi-model toolkit: cache each member's
per-clip logits on a frozen holdout (`inference.py`), learn mixing weights or use
equal/softmax/weighted-sum combiners (`optimize.py`, `combiners.py`), and write
the ensembled submission (`submit.py`). The frozen holdout split is defined by
`data/holdout_clean.json` (the only data file committed) so members are scored on
identical clips. Committed ensemble outputs:
`submissions/track_a_ensemble_*.csv`, `submissions/track_a_ens-allaxes_*.csv`.

> Ensemble gains here were small (~+0.2 pt) because members shared one backbone;
> architecture diversity (the `track_a_diverse_arch*` experiments) matters more
> than adding seeds. The end-to-end orchestration was driven by run scripts kept
> with each run's logs; the reusable building blocks are the `ensemble/` modules.

### 7.7 Launching long runs detached

```bash
PYTHONPATH=src nohup .venv/bin/python -u \
  -m smth2smth.pipelines.train \
  track=b experiment=track_b_vjepa2_vitg384_heavy_aug \
  training.checkpoint_path=checkpoints/track_b/vitg384.pt \
  hydra.run.dir=outputs/track_b_vitg384_$(date +%Y%m%d_%H%M%S) \
  > logs/track_b_vitg384_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
echo $! > logs/track_b_vitg384.pid
```

`save_last_checkpoint` (default on) writes a `*.last.pt` companion every epoch;
restart a crashed run with `training.resume_from=<...>.last.pt` to restore
weights, optimizer, scheduler, and scaler state.

---

## 8. Configuration reference (Hydra knobs)

All settings are Hydra overrides; anything in the resolved config can be set on
the command line (`group=value` to swap a file, `a.b.c=value` to set a field).

**Groups** (swap whole files):
`model={cnn_baseline,cnn_lstm,avanced_resnet50_tsm,dual_stream_rgb_diff_tsm,video_mae_vit,vjepa2,vjepa2_hf_clf}`
· `augment={none,strong,randaugment,randaugment_t,tsm_track_a,champion_ft,videomae_pretrain_t16,official_videomae_ssv2,vjepa2_heavy}`
· `train={default,champion_videomae,videomae_official_ssv2,videomae_official_ssv2_stab}`
· `pretrain={default,videomae,vjepa}` · `test={default,videomae_official_2x3,vjepa2_official_2x3}`
· `data={default,holdout_clean}` · `track={a,b}` · `experiment=<one of configs/experiment/>`

**Common fields:**

| Knob | Meaning |
| --- | --- |
| `seed`, `dataset.seed` | global / split seeds (default 42) |
| `dataset.num_frames`, `dataset.image_size` | clip length / square crop side |
| `dataset.use_official_val` | validate on real `val/` instead of an internal split |
| `dataset.official_val_holdout_ratio`, `dataset.holdout_manifest` | stratified holdout / frozen holdout JSON |
| `dataset.use_extra`, `dataset.train_extra_dir` | open-world extra train data (Track B / E6) |
| `dataset.max_samples`, `dataset.max_samples_per_class` | subsample for fast iteration |
| `dataset.time_reversal_prob`, `dataset.temporal_reversal_augment`, `dataset.class_boosting` | label-aware temporal augmentation |
| `training.{lr,epochs,batch_size,optimizer,weight_decay}` | core optimization |
| `training.{warmup_epochs,scheduler_cosine,min_lr,layer_decay}` | LR schedule / LLRD (ViT) |
| `training.{amp,amp_dtype,grad_accum_steps,max_grad_norm}` | mixed precision / grad |
| `training.{ema_enabled,ema_decay,eval_every_n_epochs,eval_ema}` | EMA + validation cadence |
| `training.{label_smoothing,videomix_mode,videomix_alpha,videomix_prob}` | loss / clip-level mixing |
| `training.{class_balance_sampler,class_balance_loss,class_balance_beta}` | class-imbalance handling |
| `training.{tta,tta_flip,tta_scales,tta_logit_adjust}` | test-time augmentation |
| `model.pretrained` | ImageNet vs symmetric normalization (and ImageNet weights for ResNet baselines) |
| `model.init_from` | warm-start trunk/encoder from an SSL checkpoint |
| `model.{freeze_backbone,lora_enabled,lora_r,lora_alpha,head_type,head_num_queries}` | probe / PEFT controls |

Defaults and inline documentation live in the YAML; `configs/data/default.yaml`
and `configs/train/default.yaml` are especially well commented.

### Adding a new model

1. Create `src/smth2smth/shared/models/<name>.py` with an `nn.Module` and a
   builder decorated `@register_model("<name>")`.
2. Import that module in `src/smth2smth/shared/models/__init__.py` so the
   registration side-effect runs.
3. Add `configs/model/<name>.yaml`, then train with `model=<name>`.

---

## 9. Submissions & validation

Format written by `pipelines.submit` / `ensemble.submit`:

```csv
video_name,predicted_class
video_1000,9
video_100051,18
```

> The competition brief illustrates the header as `video_id,label`, but the
> sample submission and the Kaggle grader accept `video_name,predicted_class`
> — that is the header on **all 80** of our scored submissions in
> `submissions/`. Class indices are integers in `[0, 33)`.

`validate_submission_csv` (in `shared/io/submission.py`) checks header, two
cells per row, integer predictions in range, and unique video names; the
submission pipeline and the end-to-end smoke test both call it. Compare two
submissions by `video_name`:

```bash
PYTHONPATH=src uv run python scripts/compare_submissions.py \
    --baseline submissions/track_a.csv --ours submissions/track_b.csv --num-classes 33
```

Other dev scripts: `scripts/plot_submission_logits_heatmap.py` (per-clip
confidence heatmap), `scripts/download_ssv2_subset_4frame.py` (build the Track-B
`train_extra/` set from full SSv2).

---

## 10. Results

Full numbers, ablations, and the per-run logbook are in `report/` (Track A:
`report/track_a.tex`; Track B: `report/track_b.tex`). Headline figures:

| Track | Approach | Score | Source |
| --- | --- | --- | --- |
| A | TSM-ResNet50 from-scratch baseline | ~37% Top-1 (Kaggle) | report |
| A | VideoMAE SSL pretrain + finetune (champion) | **55.13% Top-1 (Kaggle LB)** | `mae500-ft-f4-val90-holdout` |
| B | V-JEPA 2 ViT-L SSv2 **zero-shot** (no training) | ~45% Top-1 (Kaggle) | report §zero-shot |
| B | V-JEPA 2 ViT-L LoRA finetune | 68.09% Top-1 (Kaggle) | report logbook (Run 5) |
| B | V-JEPA 2 SSv2-finetuned + LoRA r16 (best) | see `report/track_b.tex` + `submissions/track_b_ssv2ft_lora16f_*` | — |

The exact prediction file behind every entry is in `submissions/` (named after
its experiment). Ensembling across same-backbone members added only ~+0.2 pt;
backbone/architecture diversity was the bigger lever.

---

## 11. Testing, linting, troubleshooting

```bash
uv run pytest -q                 # full suite: 284 tests
uv run pytest -q -m "not slow"   # skip the end-to-end smoke test (~few seconds)
uv run ruff check .              # lint (clean)
uv run ruff format --check .     # format (clean)
```

The `slow` test (`tests/pipelines/test_smoke_end_to_end.py`) runs the entire
train → evaluate → submit pipeline on a tiny **synthetic** dataset and validates
the produced CSV — a self-contained reproducibility check that needs no data or
GPU.

**Troubleshooting**

- *Import errors / "module smth2smth not found"* → run with `PYTHONPATH=src`
  (or use the `scripts/run_track_*.py` wrappers).
- *CUDA OOM on Track B* → lower `training.batch_size`, raise
  `training.grad_accum_steps`, keep `training.amp=true`, and shrink with
  `dataset.max_samples_per_class`.
- *V-JEPA 2 weights won't download* → check network / `$HF_HOME`; the repo id
  must be a real `facebook/vjepa2-*` checkpoint.
- *V-JEPA 2 runtime error on shapes* → `dataset.num_frames` must be even and
  `dataset.image_size` must match the encoder (256 or 384).
- *Reproduce an exact past run* → read the relevant row in
  `report/track_*.tex`, find its `experiment=` and overrides, and re-run; the
  default-off switches keep legacy behavior bit-for-bit.
```
