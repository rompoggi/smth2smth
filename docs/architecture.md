# Architecture

This document explains the package layout and how data, models, and configs flow through training, evaluation, and submission.

## Top-level package layout

```text
src/smth2smth/
  shared/
    data/        dataset + transforms (read-only data layer)
    models/      cnn_baseline, cnn_lstm, model registry
    engine/      training/eval loops, metrics
    io/          checkpoint + submission persistence
    utils/       seeding, train/val splits
  track_a/       Track A overrides (currently empty; place track-specific code here)
  track_b/       Track B overrides
  pipelines/     train, evaluate, submit (Hydra entrypoints + testable run() helpers)
```

Outside `src/`:

- `configs/` — Hydra YAML grouped by `data`, `train`, `model`, `experiment`, `track`.
- `external/prof_baseline/` — vendored read-only baseline.
- `notebooks/` — exploration & error analysis (consumes `shared/`).
- `scripts/` — track-preset wrappers and dev tools.
- `tests/` — pytest suite mirroring `src/smth2smth/` layout.

## Module dependency graph

```text
                                ┌──────────────────┐
                                │ configs/*.yaml   │
                                └─────────┬────────┘
                                          │  Hydra compose
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ pipelines/                                                                      │
│   train.py    evaluate.py    submit.py       (each: @hydra.main + run(cfg))    │
└──────┬──────────────┬──────────────┬───────────────────────────────────────────┘
       │              │              │
       ▼              ▼              ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ shared/                                                                         │
│   data/          models/          engine/             io/         utils/        │
│   ├─VideoFrame   ├─CNNBaseline    ├─train_one_epoch   ├─save_*     ├─set_seed   │
│   │ Dataset      ├─CNNLSTM        ├─evaluate_epoch    ├─load_*     └─split_*    │
│   ├─collect_*    ├─registry       ├─predict_argmax    ├─validate_* │            │
│   └─build_       │  (build_model) └─accuracy_topk     └─compare    │            │
│     transforms   │                                       helpers   │            │
└────────────────────────────────────────────────────────────────────────────────┘
```

Rules:
- `pipelines/` is the **only** layer that knows about Hydra.
- `shared/` is **Hydra-free** (engine, IO, utils, data take plain Python objects).
- `track_a/`/`track_b/` import from `shared/` only.
- `external/prof_baseline/` is **never** imported by first-party code.

## Data flow at training time

```text
  configs/config.yaml ──► Hydra ──► DictConfig
                                       │
                                       ▼
                pipelines.train.run(cfg)
                       │
       ┌───────────────┼─────────────────────────────────┐
       ▼               ▼                                 ▼
  set_seed     collect_video_samples()             build_transforms()
                       │
                       ▼
          split_train_val()  ──►  train/val sample lists
                       │
                       ▼
        VideoFrameDataset ──► DataLoader
                       │
                       ▼
                 build_model(cfg)  ──►  nn.Module on device
                       │
                       ▼
        train_one_epoch / evaluate_epoch  ──►  EpochStats
                       │
                       ▼
               save_checkpoint(path, model, cfg, extra)
```

The checkpoint format (`schema_version=1`) stores the merged config so evaluation and submission can rebuild the exact same model architecture without re-importing the training script.

## Data flow at evaluation / submission time

```text
  checkpoint file ──► load_checkpoint() ──► payload {state_dict, config, ...}
                                                │
                                                ▼
                                       cfg_from_checkpoint()
                                                │
                                                ▼
                                          build_model(cfg)
                                                │
                                                ▼
            ┌───────────────────────────────────┴────────────────────┐
            ▼                                                        ▼
   evaluate_epoch on val_dir                              predict_argmax on test_dir
            │                                                        │
            ▼                                                        ▼
        EvalReport                                       write_submission_csv()
                                                                     │
                                                                     ▼
                                                          submissions/track_*.csv
                                                                     │
                                                                     ▼
                                                       validate_submission_csv()
```

## Model registry

`shared/models/registry.py` maintains a module-level `MODEL_REGISTRY: dict[str, Callable[[DictConfig], nn.Module]]`. Each model module decorates a builder function:

```python
@register_model("cnn_baseline")
def build_cnn_baseline(cfg: DictConfig) -> nn.Module:
    return CNNBaseline(num_classes=int(cfg.model.num_classes),
                       pretrained=bool(cfg.model.pretrained))
```

Importing `smth2smth.shared.models` triggers the side-effects that populate the registry. `build_model(cfg)` then dispatches by name. Adding a new model:

1. Create `src/smth2smth/shared/models/<name>.py` with the `nn.Module` class and a `@register_model` builder.
2. Import it from `shared/models/__init__.py` so the registration runs.
3. Add `configs/model/<name>.yaml` describing the hyperparameters.
4. Optionally add `configs/experiment/<name>.yaml` to make it selectable as `experiment=<name>`.

## Tracks

Tracks are **output-scoping** overlays, not behavioural switches:

- `configs/track/a.yaml` sets `training.checkpoint_path` and `dataset.submission_output` to track-A locations.
- `configs/track/b.yaml` does the same for track B.
- The from-scratch / pretrained choice is encoded in the **experiment** file (and pinned by the track-preset scripts).

This separation means you can freely run e.g. `track=a experiment=baseline_pretrained` for ablations without polluting Track A's actual artifacts.

## Configuration composition

Hydra default groups (in `configs/config.yaml`):

```yaml
defaults:
  - model: cnn_baseline
  - data: default
  - train: default
  - experiment: baseline_pretrained
  - track: a
  - _self_
```

Each group can be overridden from the CLI:

```bash
python -m smth2smth.pipelines.train experiment=cnn_lstm track=b model.pretrained=true
```

Effective overrides land in the merged `cfg`, which is passed end-to-end to `run()`.

## Testing layout

`tests/` mirrors `src/smth2smth/`:

```text
tests/
  shared/
    data/        dataset + transforms tests
    models/      registry + forward-pass tests
    engine/      metrics + trainer tests
    io/          checkpoint + submission tests
  pipelines/
    test_smoke_end_to_end.py   (slow; full train -> evaluate -> submit on synthetic data)
  scripts/
    test_compare_submissions.py
```

The smoke test is marked `@pytest.mark.slow`. Skip with `pytest -m 'not slow'`.

## Vendored baseline

`external/prof_baseline/` is a frozen copy of the professor's starter repo, kept exclusively for reference and side-by-side comparisons. Rules:

- Treat as **read-only**. Do not edit or move files inside it.
- Do not import from it in first-party code.
- Run it (when needed) with its own `uv sync` from inside `external/prof_baseline/`.

The comparison procedure is documented in `report/parity.md`.
