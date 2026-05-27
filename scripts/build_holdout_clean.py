#!/usr/bin/env python3
"""Build frozen ``data/holdout_clean.json`` (seed 42, stratified 15% of official val).

Verifies that per-class holdout fractions match the target ratio and that train
class proportions are close to full-val proportions.

Example::

    PYTHONPATH=src uv run python scripts/build_holdout_clean.py \\
      --train-dir data/train --val-dir data/val \\
      --output data/holdout_clean.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from smth2smth.ensemble.holdout import build_and_write_holdout_clean


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build data/holdout_clean.json")
    parser.add_argument("--train-dir", type=Path, default=REPO / "data" / "train")
    parser.add_argument("--val-dir", type=Path, default=REPO / "data" / "val")
    parser.add_argument("--output", type=Path, default=REPO / "data" / "holdout_clean.json")
    parser.add_argument("--holdout-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO / "outputs" / "holdout_clean" / "distribution_report.json",
        help="Optional copy of the distribution report (manifest embeds it too).",
    )
    args = parser.parse_args()

    report = build_and_write_holdout_clean(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_path=args.output,
        holdout_ratio=args.holdout_ratio,
        split_seed=args.split_seed,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {args.output.resolve()}")
    print(f"  holdout n={report['n_holdout']} / val n={report['n_val_total']}")
    print(f"  max per-class holdout fraction error: {report['max_per_class_holdout_fraction_error']}")
    print(
        "  max |p_train - p_val| per class: "
        f"{report['max_abs_train_vs_val_proportion_error']}"
    )
    print(f"Report copy: {args.report.resolve()}")


if __name__ == "__main__":
    main()
