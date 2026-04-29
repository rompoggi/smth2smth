# External baselines

This directory contains read-only third-party code used as a reference baseline.

## Contents

- `prof_baseline/` — vendored copy of the professor's challenge starter repo
  (`What_Happens_Next_CSC_43M04_EP_2026`). Source of truth for the baseline
  results we compare against.

## Rules

- **Do not modify** files under `prof_baseline/`. Treat it as read-only.
- Do not import from `prof_baseline/` inside `src/smth2smth/`. The first-party
  code must be self-contained; ported pieces are copied and adapted into
  `src/smth2smth/shared/`.
- Run-time artifacts from baseline (checkpoints, `outputs/`, `processed_data/`)
  are git-ignored and must not be committed.

## How to run the baseline

From the repo root:

```bash
cd external/prof_baseline
uv sync  # uses its own pyproject.toml/uv.lock if you want full isolation
python src/train.py experiment=baseline_pretrained
```

See `external/prof_baseline/README.md` for full usage.
