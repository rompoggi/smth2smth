#!/usr/bin/env python3
"""Replay collected s42 FT logs to W&B as merged runs (group ft-mae-scaling)."""

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
LOG_DIR = REPO / "logs/track_a/ft_mae_scaling_s42"
PROJECT = "smth2smth-frame-ablation"
ENTITY = "romain-poggi-ecole-polytechnique"
GROUP = "ft-mae-scaling"
RUN_ID_PREFIX = "s42-replay-v2-mae"

STEP_RE = re.compile(
    r"step (\d+)/(\d+) \| avg train loss ([\d.]+) top1 ([\d.]+) top5 ([\d.]+)"
)
EPOCH_RE = re.compile(
    r"Epoch (\d+)/50 \| train loss ([\d.]+) top1 ([\d.]+) \| "
    r"val loss ([\d.]+) top1 ([\d.]+) top5 ([\d.]+) \| "
    r"ema val top1 ([\d.]+) top5 ([\d.]+)"
)
RUN_RE = re.compile(r"^meanpool-mae(\d+)-s42_\d{8}\.log$")

# First broken upload batch (auto ids, same display names).
BROKEN_RUN_IDS = [
    "gson8ohp",
    "v3migx2m",
    "bolskve3",
    "8zg8n74y",
    "eqk3acqq",
    "ijsvtkow",
    "hv6lokrb",
    "8r3jh05t",
    "pahn5dmp",
    "xyl9crqz",
]


def parse_log(path: Path) -> tuple[int, dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    """Return (steps_per_epoch, step_metrics, epoch_metrics) with deduped global steps."""
    text = path.read_text(errors="replace")
    steps_per_epoch = 5625
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


def delete_s42_replays() -> None:
    """Delete all prior s42 log-replay runs (broken duplicates included)."""
    import wandb as wandb_module

    api = wandb_module.Api()
    path = f"{ENTITY}/{PROJECT}"
    to_delete: set[str] = set(BROKEN_RUN_IDS)
    for ep in (50, 100, 150, 200, 250, 300, 350, 400, 450, 500):
        to_delete.add(f"{RUN_ID_PREFIX}{ep:03d}")
        to_delete.add(f"s42-replay-mae{ep:03d}")  # prior broken uploads

    for run in api.runs(path):
        name = run.name or ""
        tags = run.tags or []
        if run.id in to_delete:
            print(f"delete id={run.id} name={name}")
            run.delete()
            continue
        if "log-replay" in tags and name.startswith("meanpool-mae") and name.endswith("-s42"):
            print(f"delete id={run.id} name={name} (log-replay tag)")
            run.delete()
            continue
        if name.startswith("meanpool-mae") and name.endswith("-s42") and run.id.startswith("s42-replay"):
            print(f"delete id={run.id} name={name}")
            run.delete()


def upload_one(log_path: Path, *, dry_run: bool = False, offline: bool = True) -> tuple[str | None, Path | None]:
    m = RUN_RE.match(log_path.name)
    if not m:
        raise ValueError(f"unexpected log name: {log_path.name}")
    pretrain_epochs = int(m.group(1))
    run_name = f"meanpool-mae{pretrain_epochs:03d}-s42"
    run_id = f"{RUN_ID_PREFIX}{pretrain_epochs:03d}"

    steps_per_epoch, step_hits, epoch_hits = parse_log(log_path)
    combined = merge_metrics(steps_per_epoch, step_hits, epoch_hits)
    if not combined:
        raise ValueError(f"no metrics parsed from {log_path}")

    n_val = sum(1 for ep in epoch_hits if ep * steps_per_epoch in combined)
    if dry_run:
        print(
            f"{run_name}: steps={len(step_hits)} epochs={len(epoch_hits)} "
            f"combined={len(combined)} val_at_boundary={n_val} spe={steps_per_epoch}"
        )
        return None, None

    settings = Settings(init_timeout=120)
    run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        group=GROUP,
        name=run_name,
        id=run_id,
        job_type="log-replay",
        mode="offline" if offline else "online",
        tags=["seed42", "s42-collected", "log-replay"],
        config={
            "seed": 42,
            "head_type": "meanpool",
            "pretrain_epochs": pretrain_epochs,
            "experiment": "track_a_videomae_official_ssv2_ft",
            "num_frames": 4,
            "source_log": log_path.name,
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
    print(f"logged {run_name} offline ({len(epoch_hits)} val epochs, {len(combined)} steps) -> {run_dir}")
    return url, run_dir


def sync_offline_run(run_dir: Path) -> None:
    """Push one offline W&B run directory to the cloud."""
    env = os.environ.copy()
    env["WANDB_MODE"] = "online"
    cmd = [sys.executable, "-m", "wandb", "sync", str(run_dir)]
    print(f"sync: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delete-only", action="store_true", help="delete broken s42 replay runs")
    parser.add_argument("--skip-delete", action="store_true", help="skip delete step before upload")
    parser.add_argument("--sync-only", action="store_true", help="sync existing offline dirs only")
    parser.add_argument("--online", action="store_true", help="log directly online (default: offline then sync)")
    parser.add_argument("--log", type=Path, help="single log file under ft_mae_scaling_s42/")
    args = parser.parse_args()

    if args.delete_only:
        delete_s42_replays()
        return

    logs = [args.log] if args.log else sorted(LOG_DIR.glob("meanpool-mae*-s42_*.log"))

    if args.sync_only:
        dirs = sorted((REPO / "wandb").glob("offline-run-*-s42-replay-mae*"))
        for d in dirs:
            sync_offline_run(d)
        return

    if not args.dry_run and not args.skip_delete:
        delete_s42_replays()

    urls: list[str] = []
    offline_dirs: list[Path] = []
    for log_path in logs:
        url, run_dir = upload_one(log_path, dry_run=args.dry_run, offline=not args.online)
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
        manifest.write_text(
            "# meanpool-maeXXX-s42 replay runs (group ft-mae-scaling)\n"
            + "\n".join(
                f"mae{int(m.group(1)):03d} id={RUN_ID_PREFIX}{int(m.group(1)):03d} log={p.name}"
                for p in logs
                if (m := RUN_RE.match(p.name))
            )
            + "\n\n"
            + "\n".join(urls)
            + "\n"
        )
        print(f"wrote {out} and {manifest}")


if __name__ == "__main__":
    main()
