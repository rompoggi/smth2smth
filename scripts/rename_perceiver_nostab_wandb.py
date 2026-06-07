#!/usr/bin/env python3
"""Rename non-stab PerceiverQ* W&B runs to *-NoStab in group ft-mae-scaling."""

from __future__ import annotations

import argparse
import re

ENTITY = "romain-poggi-ecole-polytechnique"
PROJECT = "smth2smth-frame-ablation"
GROUP = "ft-mae-scaling"
NOSTAB = "-NoStab"
Q16_RE = re.compile(r"^perceiverQ16-mae(\d+)-s(\d+)$")
Q_RE = re.compile(r"^perceiverQ(\d+)-mae(\d+)-s(\d+)$")
PERCEIVER_RE = re.compile(r"^perceiverQ\d+-mae\d+-s\d+$")


def strip_nostab(name: str) -> str:
    return name[: -len(NOSTAB)] if name.endswith(NOSTAB) else name


def run_is_stab(run_name: str) -> bool:
    run_name = strip_nostab(run_name)
    m = Q_RE.match(run_name)
    if m:
        q, ep, seed = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if ep != 500:
            return False
        if q == 16:
            return seed == 42
        return q in (2, 4, 8, 32, 64)
    m = Q16_RE.match(run_name)
    if m:
        return int(m.group(1)) == 500 and int(m.group(2)) == 42
    return False


def should_rename(name: str) -> bool:
    base = strip_nostab(name)
    if not PERCEIVER_RE.match(base):
        return False
    if name.endswith(NOSTAB):
        return False
    return not run_is_stab(base)


def main() -> None:
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]

    import wandb

    api = wandb.Api()
    renamed: list[tuple[str, str, str]] = []
    for run in api.runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP}, per_page=300):
        cfg = run.config or {}
        old = cfg.get("training.wandb.name") or run.name
        if not should_rename(old):
            continue
        new = f"{strip_nostab(old)}{NOSTAB}"
        print(f"{'DRY' if args.dry_run else 'RENAME'}: {old} -> {new} ({run.id})")
        if not args.dry_run:
            run.name = new
            run.update()
        renamed.append((old, new, run.id))

    manifest = repo / "logs/track_a/perceiver_nostab_rename_manifest.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w") as f:
        f.write(f"# renamed {len(renamed)} W&B runs\n")
        for old, new, rid in renamed:
            f.write(f"{old}\t{new}\t{rid}\n")
    print(f"Wrote {manifest} ({len(renamed)} rows)")


if __name__ == "__main__":
    main()
