# Submission Sanity Check

This document records the manual checks that prove our submission pipeline is
correct. Update it after each meaningful change to training/inference.

> Scope: we are **not** trying to match the professor baseline metric-by-metric.
> The only things that matter are:
> 1. our submission CSV is well-formed,
> 2. the full pipeline runs end-to-end on real data,
> 3. (optional) our predictions are at least roughly comparable to the baseline.

---

## 7.1 — Real-data dry run

Goal: prove the pipeline runs cleanly on the actual dataset (not just the synthetic smoke fixture).

### Procedure

```bash
cd /Data/romain.poggi/Modal/project/smth2smth

# 1. Tiny training run on a subset (~minutes on CPU, faster on GPU).
PYTHONPATH=src uv run python -m smth2smth.pipelines.train \
    track=a \
    experiment=baseline_pretrained \
    training.epochs=1 \
    training.device=cpu \
    dataset.max_samples=200

# 2. Evaluate the produced checkpoint on the full validation split.
PYTHONPATH=src uv run python -m smth2smth.pipelines.evaluate \
    training.checkpoint_path=checkpoints/track_a/best_model.pt \
    training.device=cpu

# 3. Generate a submission CSV.
PYTHONPATH=src uv run python -m smth2smth.pipelines.submit \
    training.checkpoint_path=checkpoints/track_a/best_model.pt \
    training.device=cpu
```

### Pass criteria

- [ ] training prints `Saved new best checkpoint: ...` at least once
- [ ] reported metrics are not NaN and not exactly 0
- [ ] evaluation prints `Validation samples: N` with `N > 0`
- [ ] `submissions/track_a.csv` is created
- [ ] `validate_submission_csv` does not raise (auto-checked by 7.2 below)

### Results

_Fill in with date + key numbers once data lands._

| Date | Track | Subset size | Best val top-1 | Best val top-5 | Notes |
| --- | --- | --- | --- | --- | --- |
|     |     |     |     |     |     |

---

## 7.2 — Submission format validation

Goal: guarantee the CSV is parseable by the leaderboard parser.

### Automated check

This runs as a unit test:

```bash
uv run pytest tests/shared/io/test_submission_validator.py -v
uv run pytest tests/pipelines/test_smoke_end_to_end.py -v
```

Both must pass. The smoke test produces a synthetic CSV and runs
`validate_submission_csv` against it, asserting:

- header is exactly `video_name,predicted_class`
- all rows have two cells
- predictions are integers in `[0, num_classes)`
- video names are unique
- video name set matches the expected test split

### Manual check on real submission

Once a real `submissions/track_*.csv` exists, run:

```bash
PYTHONPATH=src uv run python -c "
from pathlib import Path
from smth2smth.shared.io.submission import validate_submission_csv
report = validate_submission_csv(Path('submissions/track_a.csv'), num_classes=33)
print(report)
"
```

### Pass criteria

- [ ] no exception raised
- [ ] `num_rows` matches the expected number of test videos
- [ ] `unique_videos == num_rows`

---

## 7.3 — (Optional) Comparison against the professor baseline

Goal: a black-box sanity check that our predictions are not wildly different
from the baseline (in case we missed a bug or mis-mapped class indices).

### Step 1 — produce both submissions

Baseline (vendored under `external/prof_baseline/`):

```bash
cd external/prof_baseline
uv sync
python src/train.py experiment=baseline_pretrained training.epochs=1
python src/create_submission.py \
    training.checkpoint_path=best_model.pt \
    dataset.submission_output=$(pwd)/submission.csv
cd ../..
```

Ours:

```bash
PYTHONPATH=src uv run python -m smth2smth.pipelines.train \
    track=b experiment=baseline_pretrained training.epochs=1
PYTHONPATH=src uv run python -m smth2smth.pipelines.submit \
    training.checkpoint_path=checkpoints/track_b/best_model.pt
```

### Step 2 — compare

```bash
PYTHONPATH=src uv run python scripts/compare_submissions.py \
    --baseline external/prof_baseline/submission.csv \
    --ours submissions/track_b.csv \
    --num-classes 33
```

Sample output::

    baseline : external/prof_baseline/submission.csv
    ours     : submissions/track_b.csv
    videos   : 1234
    agree    : 1102  (89.30%)
    disagree : 132
    top disagreement directions (baseline_class -> ours_class):
       17 -> 12   28
       12 -> 17   17
       ...

### Pass criteria (rules of thumb, not strict)

- [ ] both files validate via `validate_submission_csv`
- [ ] same set of `video_name` values (mismatch immediately fails)
- [ ] agreement rate is "reasonable":
    - identical training setup and checkpoint -> expect very high agreement (> 95%)
    - independent training runs (different seeds) -> high agreement on easy classes, lower elsewhere; sub-50% would warrant an investigation

### Results

_Fill in once both submissions are produced._

| Date | Baseline run | Our run | Videos | Agreement % | Notes |
| --- | --- | --- | --- | --- | --- |
|     |     |     |     |     |     |
