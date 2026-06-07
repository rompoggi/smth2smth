#!/usr/bin/env python3
"""Replay collected s42 Perceiver train-only Q-sweep logs to W&B (group ft-mae-scaling)."""

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
LOG_DIR = REPO / "logs/track_a/perceiver_q_s42"
PROJECT = "smth2smth-frame-ablation"
SOURCE_PROJECT = "smth2smth-diverse-heads"
ENTITY = "romain-poggi-ecole-polytechnique"
GROUP = "ft-mae-scaling"
RUN_ID_PREFIX = "s42-replay-v2-perceiverQ"
STEPS_PER_EPOCH = 5625

STEP_RE = re.compile(
    r"step (\d+)/(\d+) \| avg train loss ([\d.]+) top1 ([\d.]+) top5 ([\d.]+)"
)
EPOCH_RE = re.compile(
    r"Epoch (\d+)/50 \| train loss ([\d.]+) top1 ([\d.]+) \| "
    r"val loss ([\d.]+) top1 ([\d.]+) top5 ([\d.]+) \| "
    r"ema (?:val|holdout) top1 ([\d.]+) top5 ([\d.]+)"
)
RUN_RE = re.compile(r"^perceiverQ(\d+)-mae500-s42_\d{8}\.log$")

# Original diverse-heads run ids (seed 42, train-only).
SOURCE_RUN_IDS: dict[int, str] = {
    2: "j3pdr8gj",
    4: "yfms3io3",
    8: "mbb75n3k",
    16: "cfqjfqy1",
    32: "18la2ij5",
}

# Replay metrics from original diverse-heads W&B history (output.log files are often truncated).
USE_WANDB_HISTORY = set(SOURCE_RUN_IDS)


def parse_log(path: Path) -> tuple[int, dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    """Return (steps_per_epoch, step_metrics, epoch_metrics) with deduped global steps."""
    text = path.read_text(errors="replace")
    steps_per_epoch = STEPS_PER_EPOCH
    step_hits: dict[int, dict[str, float]] = {}
    epoch_hits: dict[int, dict[str, float]] = {}

    current_epoch = 0
    for line in text.splitlines():
        m = EPOCH_RE.search(line)
        if m:
            ep = int(m.group(1))
            current_epoch = ep
            epoch_hits[ep] = {
                "train/epoch_loss": float(m.group(2)),
                "train/epoch_top1": float(m.group(3)),
                "val/loss": float(m.group(4)),
                "val/top1": float(m.group(5)),
                "val/top5": float(m.group(6)),
                "val/ema_top1": float(m.group(7)),
                "val/ema_top5": float(m.group(8)),
                "epoch": float(ep),
            }
            continue

        m = STEP_RE.search(line)
        if m:
            step_in, spe, loss, top1, top5 = m.groups()
            steps_per_epoch = int(spe)
            step_in_i = int(step_in)
            train_epoch = current_epoch + 1
            global_step = (train_epoch - 1) * steps_per_epoch + step_in_i
            step_hits[global_step] = {
                "train/loss": float(loss),
                "train/top1": float(top1),
                "train/top5": float(top5),
            }

    return steps_per_epoch, step_hits, epoch_hits


def merge_metrics(
    steps_per_epoch: int,
    step_hits: dict[int, dict[str, float]],
    epoch_hits: dict[int, dict[str, float]],
) -> dict[int, dict[str, float]]:
    """Merge step + epoch metrics at each global step (epoch val at epoch boundaries)."""
    best_top1 = 0.0
    for ep in sorted(epoch_hits):
        best_top1 = max(best_top1, epoch_hits[ep]["val/top1"], epoch_hits[ep]["val/ema_top1"])
        epoch_hits[ep]["val/best_top1"] = best_top1

    combined: dict[int, dict[str, float]] = {}
    for step, metrics in step_hits.items():
        combined.setdefault(step, {}).update(metrics)
    for ep, metrics in epoch_hits.items():
        combined.setdefault(ep * steps_per_epoch, {}).update(metrics)
    return combined


def metrics_from_wandb_history(q: int) -> dict[int, dict[str, float]]:
    """Build merged step dict from original diverse-heads W&B history (Q4 fallback)."""
    import wandb as wandb_module

    run_id = SOURCE_RUN_IDS[q]
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
        metrics: dict[str, float] = {}
        for col in metric_cols:
            val = row[col]
            if val == val:  # skip NaN
                metrics[col] = float(val)
        if metrics:
            combined.setdefault(step, {}).update(metrics)
    return combined


def delete_prior_replays() -> None:
    """Delete prior perceiverQ s42 log-replay runs in frame-ablation."""
    import wandb as wandb_module

    api = wandb_module.Api()
    path = f"{ENTITY}/{PROJECT}"
    for q in SOURCE_RUN_IDS:
        stale_ids = {
            f"{RUN_ID_PREFIX}{q}-mae500",
            f"s42-replay-perceiverQ{q}-mae500",
        }
        for run in api.runs(path):
            name = run.name or ""
            if run.id in stale_ids:
                print(f"delete id={run.id} name={name}")
                run.delete()
                continue
            if name == f"perceiverQ{q}-mae500-s42" and "log-replay" in (run.tags or []):
                print(f"delete id={run.id} name={name} (log-replay tag)")
                run.delete()


def log_combined(
    combined: dict[int, dict[str, float]],
    *,
    run_name: str,
    run_id: str,
    q: int,
    source_log: str,
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
        tags=["seed42", "s42-collected", "log-replay", "train-only", "perceiver"],
        config={
            "seed": 42,
            "head_type": f"perceiverQ{q}",
            "num_queries": q,
            "pretrain_epochs": 500,
            "include_val_in_train": False,
            "experiment": "track_a_diverse_arch2_perceiver",
            "num_frames": 4,
            "source_log": source_log,
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
    n_val = sum(1 for m in combined.values() if "val/top1" in m)
    print(f"logged {run_name} ({n_val} val points, {len(combined)} steps) -> {run_dir}")
    return url, run_dir


def upload_one(
    log_path: Path | None,
    q: int,
    *,
    dry_run: bool = False,
    offline: bool = True,
) -> tuple[str | None, Path | None]:
    run_name = f"perceiverQ{q}-mae500-s42"
    run_id = f"{RUN_ID_PREFIX}{q}-mae500"
    source_run_id = SOURCE_RUN_IDS[q]

    if q in USE_WANDB_HISTORY:
        combined = metrics_from_wandb_history(q)
        source_log = f"wandb-history:{source_run_id}"
        n_epochs = sum(1 for m in combined.values() if "epoch" in m and "val/top1" in m)
        if dry_run:
            print(
                f"{run_name}: wandb-history steps={len(combined)} val_epochs~={n_epochs} "
                f"source={source_run_id}"
            )
            return None, None
    else:
        if log_path is None:
            raise ValueError(f"log required for Q{q}")
        steps_per_epoch, step_hits, epoch_hits = parse_log(log_path)
        combined = merge_metrics(steps_per_epoch, step_hits, epoch_hits)
        source_log = log_path.name
        if dry_run:
            n_val = sum(1 for ep in epoch_hits if ep * steps_per_epoch in combined)
            print(
                f"{run_name}: steps={len(step_hits)} epochs={len(epoch_hits)} "
                f"combined={len(combined)} val_at_boundary={n_val} spe={steps_per_epoch}"
            )
            return None, None

    if not combined:
        raise ValueError(f"no metrics for {run_name}")

    url, run_dir = log_combined(
        combined,
        run_name=run_name,
        run_id=run_id,
        q=q,
        source_log=source_log,
        source_run_id=source_run_id,
        offline=offline,
    )
    return url, run_dir


def sync_offline_run(run_dir: Path) -> None:
    env = os.environ.copy()
    env["WANDB_MODE"] = "online"
    cmd = [sys.executable, "-m", "wandb", "sync", str(run_dir)]
    print(f"sync: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delete-only", action="store_true")
    parser.add_argument("--skip-delete", action="store_true")
    parser.add_argument("--sync-only", action="store_true")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--log", type=Path, help="single collected log under perceiver_q_s42/")
    parser.add_argument("--q", type=int, choices=sorted(SOURCE_RUN_IDS), help="single Q value")
    args = parser.parse_args()

    if args.delete_only:
        delete_prior_replays()
        return

    if args.sync_only:
        for d in sorted((REPO / "wandb").glob(f"offline-run-*-{RUN_ID_PREFIX}*")):
            sync_offline_run(d)
        return

    if args.q:
        qs = [args.q]
        logs = {}
        if args.q not in USE_WANDB_HISTORY:
            pat = f"perceiverQ{args.q}-mae500-s42_*.log"
            found = sorted(LOG_DIR.glob(pat))
            if not found and args.log:
                found = [args.log]
            if not found:
                raise SystemExit(f"no log for Q{args.q} under {LOG_DIR}")
            logs[args.q] = found[0]
    elif args.log:
        m = RUN_RE.match(args.log.name)
        if not m:
            raise SystemExit(f"unexpected log name: {args.log.name}")
        qs = [int(m.group(1))]
        logs = {qs[0]: args.log}
    else:
        qs = sorted(SOURCE_RUN_IDS)
        logs = {}
        for q in qs:
            if q in USE_WANDB_HISTORY:
                continue
            found = sorted(LOG_DIR.glob(f"perceiverQ{q}-mae500-s42_*.log"))
            if found:
                logs[q] = found[0]

    if not args.dry_run and not args.skip_delete:
        delete_prior_replays()

    urls: list[str] = []
    offline_dirs: list[Path] = []
    for q in qs:
        url, run_dir = upload_one(
            logs.get(q),
            q,
            dry_run=args.dry_run,
            offline=not args.online,
        )
        if url:
            urls.append(url)
        if run_dir:
            offline_dirs.append(run_dir)

    if offline_dirs and not args.dry_run and not args.online:
        for run_dir in offline_dirs:
            sync_offline_run(run_dir)

    if urls:
        out = LOG_DIR / "wandb_replay_urls.txt"
        out.write_text("\n".join(urls) + "\n")
        manifest = LOG_DIR / "wandb_replay_manifest.txt"
        lines = [
            "# perceiverQ*-mae500-s42 replay runs (group ft-mae-scaling)",
            "# source project: smth2smth-diverse-heads",
        ]
        for q in qs:
            rid = f"{RUN_ID_PREFIX}{q}-mae500"
            src = SOURCE_RUN_IDS[q]
            log_name = logs[q].name if q in logs else f"wandb-history:{src}"
            lines.append(f"Q{q} id={rid} source={src} log={log_name}")
        lines.extend(["", *urls])
        manifest.write_text("\n".join(lines) + "\n")
        print(f"wrote {out} and {manifest}")


if __name__ == "__main__":
    main()
