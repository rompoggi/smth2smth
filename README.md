# smth2smth — Something-Something v2 Video Classification

Project for the **CSC_43M04_EP — Modal d'informatique — Deep Learning in Computer Vision** challenge ("What_Happens_Next?"), built around the Something-Something dataset and submitted as a Kaggle competition.

The repository is organised as a single Python package (`smth2smth`) that supports two challenge tracks from one codebase, with a clean separation between reusable building blocks and track-specific overrides.

## Group members

- Romain Poggi
- Thomas Turkieh

## Challenge tracks

| | Track A — Closed World | Track B — Open World |
| --- | --- | --- |
| Pretrained weights | not allowed | allowed |
| External data | not allowed | allowed |
| Default experiment | `baseline_from_scratch` | `baseline_pretrained` |
| Output paths | `checkpoints/track_a/`, `submissions/track_a.csv` | `checkpoints/track_b/`, `submissions/track_b.csv` |

## Repository layout

```text
smth2smth/
  configs/                 # Hydra YAML
    config.yaml            # root composition
    data/, train/, model/, experiment/, track/
  data/                    # train/, val/, test/ (gitignored)
  external/
    prof_baseline/         # vendored read-only baseline
  notebooks/               # exploration & error analysis (5 stubs)
  report/
    parity.md              # submission sanity check & comparison procedure
  scripts/
    download_data.py
    run_track_a.py         # track-preset wrapper
    run_track_b.py
    compare_submissions.py # diff two submission CSVs
  src/smth2smth/
    shared/
      data/                # dataset + transforms
      models/              # cnn_baseline, cnn_lstm, registry
      engine/              # trainer, evaluator, metrics
      io/                  # checkpoint + submission IO
      utils/               # seed, splits
    track_a/, track_b/     # track-specific overrides (currently empty)
    pipelines/             # train, evaluate, submit (Hydra entrypoints)
  tests/                   # pytest suite (81 tests)
  pyproject.toml           # uv + pytest + ruff config
```

See `docs/architecture.md` for module dependencies and the complete data flow.

## Quick start

### 1. Environment

```bash
uv venv --python 3.12.9
uv sync
```

### 2. Data

Place the dataset (already frame-extracted) under `data/`:

```text
data/
  train/<class>/<video>/frame_*.jpg
  val/<class>/<video>/frame_*.jpg
  test/<video>/frame_*.jpg
```

If your data already lives elsewhere on disk, symlink instead of copying:

```bash
ln -s /absolute/path/to/frames data
```

The download helper (kept for reference, may need updating):

```bash
uv run scripts/download_data.py
```

### 3. Train, evaluate, submit

The package is not installed (`[tool.uv] package = false`), so run with `PYTHONPATH=src`:

```bash
# Track A (from scratch)
PYTHONPATH=src uv run python -m smth2smth.pipelines.train track=a experiment=baseline_from_scratch
PYTHONPATH=src uv run python -m smth2smth.pipelines.evaluate \
    training.checkpoint_path=checkpoints/track_a/best_model.pt
PYTHONPATH=src uv run python -m smth2smth.pipelines.submit \
    training.checkpoint_path=checkpoints/track_a/best_model.pt

# Track B (pretrained)
PYTHONPATH=src uv run python -m smth2smth.pipelines.train track=b experiment=baseline_pretrained
```

Or use the track-preset scripts which pin the right `track=` and `experiment=` overrides:

```bash
python scripts/run_track_a.py training.epochs=20
python scripts/run_track_b.py training.epochs=20 training.lr=0.0005
```

Any Hydra override works on top:

```bash
python scripts/run_track_b.py \
    training.batch_size=16 \
    dataset.num_frames=12 \
    model.pretrained=true
```

## Configuration

All tunables live in YAML under `configs/`, loaded by Hydra. Highlights:

- `configs/config.yaml` — root composition (`num_classes`, `seed`, default groups).
- `configs/data/default.yaml` — `dataset.{root,train_dir,val_dir,test_dir}`, `num_frames`, `image_size`, `val_ratio`.
- `configs/train/default.yaml` — `batch_size`, `lr`, `epochs`, `device`, `checkpoint_path`.
- `configs/model/{cnn_baseline,cnn_lstm}.yaml` — architecture hyperparameters.
- `configs/experiment/{baseline_from_scratch,baseline_pretrained}.yaml` — preset experiments.
- `configs/track/{a,b}.yaml` — track output paths and metadata.

Inspect the resolved config without running:

```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.train --cfg job --resolve track=b
```

## Running models

This section gives concrete commands for the most common runs.

| Run | Track | Model | Init | Augmentation |
| --- | --- | --- | --- | --- |
| Baseline (Track A) | A (closed) | `cnn_baseline` (ResNet-18 + temporal avg pool) | random | hflip (default) |
| Baseline (Track B) | B (open) | `cnn_baseline` | ImageNet-pretrained | hflip (default) |
| CNN-LSTM | any | `cnn_lstm` (ResNet-18 frame encoder + LSTM) | configurable | hflip (default) |
| Stronger aug | any | any | any | hflip + color jitter + random crop (`augment=strong`) |

Every command below assumes you've completed the **Quick start** above (env + data in place).

### Baseline — from scratch (Track A)

The "no pretrained weights, no external data" entry. Trains a frame-level CNN (ResNet-18) plus temporal average pooling, with random initialization.

```bash
python scripts/run_track_a.py training.epochs=20
```

Expanded form (what the script does):

```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.train \
    track=a \
    experiment=baseline_from_scratch \
    training.epochs=20
```

Outputs:
- checkpoint → `checkpoints/track_a/best_model.pt`
- training logs → printed to stdout (and Hydra's per-run `outputs/<date>/<time>/` dir)

Then evaluate and produce a submission:

```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.evaluate \
    track=a training.checkpoint_path=checkpoints/track_a/best_model.pt
PYTHONPATH=src uv run python -m smth2smth.pipelines.submit \
    track=a training.checkpoint_path=checkpoints/track_a/best_model.pt
# -> submissions/track_a.csv
```

### Baseline — pretrained (Track B)

Same architecture, ImageNet weights for the ResNet-18 backbone:

```bash
python scripts/run_track_b.py training.epochs=20
```

Expanded:

```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.train \
    track=b \
    experiment=baseline_pretrained \
    training.epochs=20
```

### CNN-LSTM

Replaces the temporal average pooling with an LSTM over the per-frame embeddings. Selected by composing the `cnn_lstm` model:

```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.train \
    track=b \
    experiment=baseline_pretrained \
    model=cnn_lstm \
    model.lstm_hidden_size=256 \
    training.epochs=20
```

For a from-scratch CNN-LSTM (Track A), pin `model.pretrained=false`:

```bash
python scripts/run_track_a.py model=cnn_lstm model.pretrained=false training.epochs=20
```

### Default data augmentation

The training pipeline already applies a light, per-frame augmentation policy via `shared/data/transforms.py`. When `is_training=True`, the transform is:

```text
Resize((image_size, image_size))
RandomHorizontalFlip()
ToTensor()
Normalize(mean, std)         # ImageNet stats if pretrained, symmetric (0.5, 0.5, 0.5) otherwise
```

So the baseline commands above **already train with basic augmentation** (random horizontal flip). At evaluation/submission time the pipeline calls `build_transforms(..., is_training=False)`, which drops the flip — predictions are deterministic.

### Stronger data augmentation (no code edit needed)

Augmentation is now controlled by Hydra config group `augment`:

- `augment=none` (default): resize + random horizontal flip
- `augment=strong`: resize(224+32) + random crop + random horizontal flip + color jitter

Examples:

```bash
# Track A with strong augmentation
python scripts/run_track_a.py augment=strong training.epochs=20

# Track B with strong augmentation and lower learning rate
python scripts/run_track_b.py augment=strong training.lr=0.0001 training.epochs=20
```

Under the hood, evaluation/submission load the augment policy from the checkpoint config so pre-processing remains shape-compatible with training (e.g. crop-padding + center-crop behavior).

## How to run experiments (Hydra patterns)

Common knobs:

- `track={a,b}`
- `experiment={baseline_from_scratch,baseline_pretrained}`
- `model={cnn_baseline,cnn_lstm}`
- `augment={none,strong}`
- `training.lr=...`, `training.epochs=...`, `training.batch_size=...`
- `dataset.num_frames=...`, `dataset.image_size=...`

Single run:

```bash
python scripts/run_track_b.py augment=strong model=cnn_lstm training.lr=0.0001 training.epochs=20
```

LR sweep (`--multirun` / `-m`):

```bash
python scripts/run_track_b.py -m augment=none,strong training.lr=5e-4,1e-4,5e-5 training.epochs=20 \
  'training.checkpoint_path=${hydra:runtime.cwd}/checkpoints/runs/track_b_aug-${augment.name}_lr${training.lr}.pt'
```

Inspect the resolved config before launching:

```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.train --cfg job --resolve \
    track=b experiment=baseline_pretrained augment=strong model=cnn_lstm
```

## Add a new model + register it

1. Create a model module, e.g. `src/smth2smth/shared/models/my_model.py`, with:
   - a `torch.nn.Module` class
   - a builder decorated with `@register_model("my_model")`
2. Import the module in `src/smth2smth/shared/models/__init__.py` so registration side-effects run.
3. Add `configs/model/my_model.yaml` for model hyperparameters.
4. Train by selecting it from Hydra: `model=my_model`.

Minimal builder pattern:

```python
@register_model("my_model")
def build_my_model(cfg: DictConfig) -> nn.Module:
    return MyModel(
        num_classes=int(cfg.model.num_classes),
        hidden_size=int(cfg.model.hidden_size),
    )
```

Minimal config:

```yaml
# configs/model/my_model.yaml
model:
  name: my_model
  num_classes: ${num_classes}
  hidden_size: 256
```

Sanity check registration:

```bash
PYTHONPATH=src uv run python -c "from smth2smth.shared.models import list_registered_models; print(list_registered_models())"
```

## Check accuracy (Top-1 / Top-5)

Use the evaluation pipeline on a checkpoint:

```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.evaluate \
    track=a training.checkpoint_path=checkpoints/track_a/best_model.pt
```

It prints:

- `Validation samples: ...`
- `Top-1 accuracy: ...`
- `Top-5 accuracy: ...`
- `Loss: ...`

### Inspecting & debugging a run

Resolve and print the merged config without launching training:

```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.train --cfg job --resolve track=b model=cnn_lstm
```

Run on CPU (e.g. for a quick sanity check, or on a node without GPU):

```bash
python scripts/run_track_a.py training.device=cpu training.epochs=1 dataset.max_samples=64
```

Limit dataset size for fast iteration on the dev loop:

```bash
python scripts/run_track_b.py dataset.max_samples=512 training.epochs=2
```

## Augmentation policy

Per-frame augmentation is applied **online** in `shared/data/transforms.py`. No on-disk augmented copies; this avoids exploding the 2.7 GB dataset and keeps fresh randomness each epoch.

When/if offline augmentation becomes worthwhile (e.g. for very expensive transforms), the `dataset.augmented_dirs` config field is reserved for that hook.

## Notebooks

Five exploration notebooks under `notebooks/` consume the package and produce visual artifacts:

| Notebook | Purpose |
| --- | --- |
| `01_dataset_overview.ipynb` | Class distribution, frames per video, video previews. |
| `02_preprocessing_check.ipynb` | Raw vs transformed frames, tensor shape/dtype/range. |
| `03_train_curves.ipynb` | Loss/top-1/top-5 across epochs and runs. |
| `04_error_analysis.ipynb` | Confusion matrix, hardest examples (uses checkpoints). |
| `05_track_comparison.ipynb` | Track A vs B summary + per-class delta. |

Launch with the project's venv:

```bash
uv run jupyter lab
```

If you need a globally-discoverable kernel:

```bash
uv run python -m ipykernel install --user --name smth2smth --display-name "Python (smth2smth)"
```

Notebook conventions:
- Code lives in `src/smth2smth/`. Notebooks **import** from it; they don't define logic.
- Each notebook starts with `%load_ext autoreload` + `%autoreload 2` and adds `<repo>/src` to `sys.path`.
- Restart-and-run-all before sharing.

## Testing & linting

```bash
uv run pytest -q              # 81 tests, ~3 s on CPU
uv run pytest -m "not slow"   # skip the end-to-end smoke test
uv run ruff check .
uv run ruff format --check .
```

The end-to-end smoke test runs the full train -> evaluate -> submit pipeline on a tiny synthetic dataset (`tests/pipelines/test_smoke_end_to_end.py`).

## Submission validation & comparison

Every smoke-test run validates the produced CSV via `validate_submission_csv` (header, integer predictions, value range, unique names).

To compare two submissions (e.g. baseline vs ours):

```bash
PYTHONPATH=src uv run python scripts/compare_submissions.py \
    --baseline external/prof_baseline/submission.csv \
    --ours submissions/track_b.csv \
    --num-classes 33
```

See `report/parity.md` for the full sanity-check procedure.

## Vendored baseline

`external/prof_baseline/` is a **read-only** copy of the professor's challenge starter repo. Use it as a reference benchmark; do not modify it. To run the baseline:

```bash
cd external/prof_baseline
uv sync
python src/train.py experiment=baseline_pretrained
```

## Tracks results

Filled in after training:

| Track | Best val top-1 | Best val top-5 | Notes |
| --- | --- | --- | --- |
| A (closed) | _TBD_ | _TBD_ |  |
| B (open)   | _TBD_ | _TBD_ |  |
