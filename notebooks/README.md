# Notebooks

Exploration and error-analysis notebooks. They consume `smth2smth.shared.*` and produce visual artifacts for the report.

## Contents

| File | Purpose |
| --- | --- |
| `01_dataset_overview.ipynb` | Class distribution, frames per video, sample previews. |
| `02_preprocessing_check.ipynb` | Raw vs transformed frames, tensor shape/dtype/range. |
| `03_train_curves.ipynb` | Loss/top-1/top-5 across epochs and runs. |
| `04_error_analysis.ipynb` | Confusion matrix, hardest examples (uses checkpoints). |
| `05_track_comparison.ipynb` | Track A vs B summary + per-class delta. |

## Setup

```bash
uv run jupyter lab
```

Or register a globally-discoverable kernel for use from any Jupyter installation:

```bash
uv run python -m ipykernel install --user --name smth2smth --display-name "Python (smth2smth)"
```

Each notebook starts with a setup cell that:

1. Loads `%autoreload 2` so edits to `src/smth2smth/` are picked up without restarting the kernel.
2. Adds `<repo_root>/src` to `sys.path` so the package is importable without an editable install.
3. Imports only from `smth2smth.shared.*`.

## Hygiene rules

- **Code lives in `src/smth2smth/`. Notebooks import; they don't define logic.**
  - If you find yourself writing a function in a notebook, move it to `shared/utils/` (or wherever appropriate) and re-import it.
- **Restart-and-run-all** before saving anything you'll share or commit. This kills hidden-state bugs from out-of-order execution.
- **No tuning loops in notebooks.** Tuning belongs in `pipelines/`. Notebooks visualize and analyze the **results** of those runs.
- **Strip outputs before committing** if the notebook contains large images:
  ```bash
  uv run nbstripout notebooks/*.ipynb   # only if nbstripout is installed
  ```
  Alternatively, use `jupytext` to pair `.ipynb` with a versionable `.py:percent` file.

## Common pitfalls with `%autoreload 2`

- Won't pick up changes to **already-instantiated objects** (their `__class__` may be stale). Re-create the object after editing its class.
- Won't refresh **already-bound names** when you edit `__init__.py` re-exports. A targeted `from importlib import reload; reload(module)` helps.
- Slight per-cell overhead — usually unnoticeable.

## Adding a new notebook

1. Pick the next number prefix (`06_*.ipynb`).
2. Copy the setup cell pattern from any existing notebook.
3. Keep one notebook = one question.
4. Add an entry to the table at the top of this file.
