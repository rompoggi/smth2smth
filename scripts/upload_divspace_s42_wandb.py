#!/usr/bin/env python3
"""Replay arch3 divided-ST train-only logs to W&B (group ft-mae-scaling)."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import wandb
from wandb.sdk.wandb_settings import Settings

REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "logs/track_a/divspace_s42"
PROJECT = "smth2smth-frame-ablation"
SOURCE_PROJECT = "smth2smth-diverse-heads"
ENTITY = "romain-poggi-ecole-polytechnique"
GROUP = "ft-mae-scaling"
RUN_ID_PREFIX = "s42-replay-v2-DivSpaceTimeK"

# Original diverse-heads train-only run ids (seed 42).
SOURCE_RUN_IDS: dict[int, str] = {
    3: "hckni98d",
    6: "oz731qwd",
    9: "zcjqthxh",
}

STEP_RE = re.compile(
    r"step (\d+)/(\d+) \| avg train loss ([\d.]+) top1 ([\d.]+) top5 ([\d.]+)"
)
EPOCH_RE = re.compile(
    r"Epoch (\d+)/50 \| train loss ([\d.]+) top1 ([\d.]+) \| "
    r"val loss ([\d.]+) top1 ([\d.]+) top5 ([\d.]+) \| "
    r"ema (?:val|holdout) top1 ([\d.]+) top5 ([\d.]+)"
)


def metrics_from_wandb_history(k: int, max_step: int | None = None) -> dict[int, dict[str, float]]:
    import wandb as wandb_module

    run_id = SOURCE_RUN_IDS[k]
    api = wandb_module.Api()
    run = api.run(f"{ENTITY}/{SOURCE_PROJECT}/{run_id}")
    hist = run.history(samples=100_000)
    metric_cols = [
        c
        for c in hist.columns
        if c.startswith(("train/", "val/", "epoch"))
        and not c.endswith("__MIN")
        and not c.endswith("__MAX")
        and c != "_timestamp"
    ]
    combined: dict[int, dict[str, float]] = {}
    for _, row in hist.iterrows():
        step = int(row["_step"])
        if max_step is not None and step > max_step:
            continue
        metrics: dict[str, float] = {}
        for col in metric_cols:
            val = row[col]
            if val == val:
                metrics[col] = float(val)
        if metrics:
            combined.setdefault(step, {}).update(metrics)
    return combined


def log_combined(
    combined: dict[int, dict[str, float]],
    *,
    run_name: str,
    run_id: str,
    k: int,
    source_run_id: str,
    offline: bool,
) -> tuple[str, Path | None]:
    settings = Settings(init_timeout=120)
    run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        group=GROUP,
        name=run_name,
        id=run_id,
        job_type="log-replay",
        mode="offline" if offline else "online",
        tags=["seed42", "s42-collected", "log-replay", "train-only", "divided_st"],
        config={
            "seed": 42,
            "head_type": f"DivSpaceTimeK{k}",
            "temporal_layers": k,
            "pretrain_epochs": 500,
            "include_val_in_train": False,
            "experiment": "track_a_diverse_arch3_divided_st_stab",
            "num_frames": 4,
            "source_wandb_project": SOURCE_PROJECT,
            "source_wandb_run_id": source_run_id,
            "replay": True,
        },
        settings=settings,
    )
    batch: list[tuple[int, dict[str, float]]] = []
    for step in sorted(combined):
        batch.append((step, combined[step]))
        if len(batch) >= 200:
            for s, metrics in batch:
                run.log(metrics, step=s, commit=False)
            run.log({}, commit=True)
            batch.clear()
    if batch:
        for s, metrics in batch:
            run.log(metrics, step=s, commit=False)
        run.log({}, commit=True)
    run_dir = Path(run.dir).parent if run.dir else None
    url = f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run_id}"
    wandb.finish()
    print(f"logged {run_name} ({len(combined)} steps) -> {run_dir}")
    return url, run_dir


def sync_offline_run(run_dir: Path) -> None:
    env = os.environ.copy()
    env["WANDB_MODE"] = "online"
    cmd = [sys.executable, "-m", "wandb", "sync", str(run_dir)]
    print(f"sync: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def upload_one(k: int, *, max_step: int | None = None, offline: bool = True) -> tuple[str, str]:
    run_name = f"DivSpaceTimeK{k}-mae500-s42"
    run_id = f"{RUN_ID_PREFIX}{k}-mae500"
    combined = metrics_from_wandb_history(k, max_step=max_step)
    if not combined:
        raise ValueError(f"no metrics for K={k}")
    url, run_dir = log_combined(
        combined,
        run_name=run_name,
        run_id=run_id,
        k=k,
        source_run_id=SOURCE_RUN_IDS[k],
        offline=offline,
    )
    if offline and run_dir is not None:
        sync_offline_run(run_dir)
    return url, run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, choices=sorted(SOURCE_RUN_IDS), action="append")
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    ks = args.k if args.k else sorted(SOURCE_RUN_IDS)
    urls: list[str] = []
    for k in ks:
        url, rid = upload_one(k, max_step=args.max_step, offline=not args.online)
        urls.append(f"K{k} {url} id={rid}")
    manifest = LOG_DIR / "wandb_replay_urls.txt"
    manifest.write_text("\n".join(urls) + "\n")
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
