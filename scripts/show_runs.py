#!/usr/bin/env python3
"""Inspect every training run by reading the checkpoints under ``checkpoints/runs/``.

Each checkpoint stores its merged Hydra config plus ``extra`` metrics
(``val_top1``, ``val_top5``, ``val_loss``, ``epoch``). This script crawls a
directory of checkpoints and prints a sorted leaderboard, optionally filtered
by track.

Usage::

    python scripts/show_runs.py
    python scripts/show_runs.py --track a
    python scripts/show_runs.py --runs-dir checkpoints/runs --sort top5

Returns exit code 0 even if no runs are found (so it's safe to call from cron
or another script).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _load_run_summary(checkpoint_path: Path) -> dict[str, Any] | None:
    """Load a single checkpoint and extract the columns used in the leaderboard.

    Args:
        checkpoint_path: Path to a ``.pt`` file written by
            :func:`smth2smth.shared.io.checkpoints.save_checkpoint`.

    Returns:
        A dict ready to be rendered as a row, or ``None`` if the file can't
        be parsed (e.g. mid-write, corrupted).
    """
    try:
        from smth2smth.shared.io.checkpoints import load_checkpoint
    except Exception as exc:
        raise SystemExit(f"Could not import smth2smth: {exc}. Run from the repo root.") from exc

    try:
        payload = load_checkpoint(checkpoint_path, map_location="cpu")
    except Exception as exc:
        print(f"  [skip] {checkpoint_path.name}: {exc}", file=sys.stderr)
        return None

    cfg = payload.get("config") or {}
    extra = payload.get("extra") or {}

    track = (cfg.get("track") or {}).get("name", "?")
    augment = (cfg.get("augment") or {}).get("name", "?")
    model = (cfg.get("model") or {}).get("name", "?")
    pretrained = bool((cfg.get("model") or {}).get("pretrained", False))
    training = cfg.get("training") or {}
    lr = float(training.get("lr", float("nan")))
    epochs_total = int(training.get("epochs", 0))

    return {
        "file": checkpoint_path.name,
        "track": track,
        "model": model,
        "pre": "Y" if pretrained else "N",
        "aug": augment,
        "lr": lr,
        "best_epoch": int(extra.get("epoch", 0)),
        "of": epochs_total,
        "top1": float(extra.get("val_top1", float("nan"))),
        "top5": float(extra.get("val_top5", float("nan"))),
        "val_loss": float(extra.get("val_loss", float("nan"))),
    }


def _format_table(rows: list[dict[str, Any]]) -> str:
    """Render rows as a fixed-width ASCII table sorted by descending val top-1."""
    if not rows:
        return "(no runs found)"

    headers = [
        ("track", "track"),
        ("model", "model"),
        ("pre", "pre"),
        ("aug", "aug"),
        ("lr", "lr"),
        ("best_epoch", "best@"),
        ("of", "/of"),
        ("top1", "top1"),
        ("top5", "top5"),
        ("val_loss", "val_loss"),
        ("file", "file"),
    ]

    def fmt(row: dict[str, Any], key: str) -> str:
        v = row[key]
        if key == "lr":
            return f"{v:.0e}" if v == v else "nan"
        if key in ("top1", "top5"):
            return f"{v:.4f}" if v == v else "nan"
        if key == "val_loss":
            return f"{v:.3f}" if v == v else "nan"
        return str(v)

    cols = [(key, label, max(len(label), max(len(fmt(r, key)) for r in rows))) for key, label in headers]

    sep = "  "
    header_line = sep.join(label.ljust(width) for _, label, width in cols)
    rule = "-" * len(header_line)
    lines = [header_line, rule]
    for row in rows:
        lines.append(sep.join(fmt(row, key).ljust(width) for key, _, width in cols))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "runs",
        help="Directory holding *.pt checkpoints (default: checkpoints/runs/).",
    )
    parser.add_argument(
        "--track",
        choices=["a", "b"],
        default=None,
        help="Filter to a single track.",
    )
    parser.add_argument(
        "--sort",
        choices=["top1", "top5", "val_loss", "lr", "file"],
        default="top1",
        help="Column to sort by (default: top1, descending; val_loss ascending).",
    )
    args = parser.parse_args()

    runs_dir: Path = args.runs_dir
    if not runs_dir.is_dir():
        print(f"No runs directory: {runs_dir}", file=sys.stderr)
        return 0

    checkpoints = sorted(runs_dir.glob("*.pt"))
    if not checkpoints:
        print(f"No *.pt files under {runs_dir}", file=sys.stderr)
        return 0

    rows = [s for s in (_load_run_summary(p) for p in checkpoints) if s is not None]
    if args.track is not None:
        rows = [r for r in rows if r["track"] == args.track]

    descending = args.sort != "val_loss"
    rows.sort(key=lambda r: r[args.sort], reverse=descending)

    print(f"Runs under: {runs_dir}")
    print(f"Found {len(rows)} checkpoint(s). Sorted by {args.sort} ({'desc' if descending else 'asc'}).\n")
    print(_format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
