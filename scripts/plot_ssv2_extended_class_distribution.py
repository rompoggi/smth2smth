#!/usr/bin/env python3
"""Compare per-class clip counts: local train+val vs local + SSv2 extras (part 00).

Reads ``outputs/ssv2_extended/local_vs_ssv2_part00_overlap.csv`` (from overlap analysis)
and writes a stacked bar chart (one bar per class: local bottom, extras top)
with seaborn blue colors.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/plot_ssv2_extended_class_distribution.py
    PYTHONPATH=src .venv/bin/python scripts/plot_ssv2_extended_class_distribution.py \\
        --csv outputs/ssv2_extended/local_vs_ssv2_part00_overlap.csv \\
        --output report/figures/ssv2_extended_class_distribution_part00.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _short_class_label(folder: str) -> str:
    """``000_Closing_something`` → ``000 Closing``."""
    parts = folder.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        name = parts[1].replace("_", " ")
        if len(name) > 22:
            name = name[:20] + "…"
        return f"{parts[0]} {name}"
    return folder[:24]


def plot_distribution(
    rows: list[dict[str, str]],
    *,
    output: Path,
    title_suffix: str,
    extra_label: str,
) -> None:
    """Draw stacked bars: local (bottom) + SSv2 extras (top) per class."""
    labels = [_short_class_label(r["class"]) for r in rows]
    local = np.array([int(r["local_total"]) for r in rows], dtype=np.float64)
    extra = np.array([int(r["extra_vs_local"]) for r in rows], dtype=np.float64)

    blues = sns.color_palette("Blues", n_colors=4)
    color_local = blues[1]
    color_extra = blues[3]

    n = len(labels)
    x = np.arange(n)
    width = 0.72

    fig, ax = plt.subplots(figsize=(max(14, n * 0.42), 6))
    ax.bar(
        x,
        local,
        width,
        label="Local (train+val)",
        color=color_local,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.bar(
        x,
        extra,
        width,
        bottom=local,
        label=extra_label,
        color=color_extra,
        edgecolor="white",
        linewidth=0.4,
    )

    ax.set_ylabel("Clip count")
    ax.set_xlabel("Class")
    ax.set_title(f"Per-class clip distribution — local + extended{title_suffix}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)

    total_local = int(local.sum())
    total_extra = int(extra.sum())
    ax.text(
        0.01,
        0.98,
        f"Local total: {total_local:,}  |  +extras: {total_extra:,}  →  {total_local + total_extra:,}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="#333333",
    )

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "outputs/ssv2_extended/local_vs_ssv2_part00_overlap.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "report/figures/ssv2_extended_class_distribution_part00.png",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Optional PDF copy (same stem as --output if omitted).",
    )
    parser.add_argument(
        "--title-suffix",
        default=None,
        help="Chart title suffix (default: inferred from CSV stem).",
    )
    parser.add_argument(
        "--extra-label",
        default=None,
        help="Legend label for stacked top segment (default: inferred from CSV stem).",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        print(f"Missing CSV: {args.csv}")
        return 1

    rows = _load_rows(args.csv)
    if not rows:
        print("CSV is empty.")
        return 1

    stem = args.csv.stem
    if args.title_suffix is not None:
        suffix = args.title_suffix
    elif "part00" in stem:
        suffix = " (part 00 on disk)"
    elif "full" in stem:
        suffix = " (full SSv2 extract, 32-class pruned)"
    else:
        suffix = ""
    if args.extra_label is not None:
        extra_label = args.extra_label
    elif "part00" in stem:
        extra_label = "SSv2 extras (part 00)"
    elif "full" in stem:
        extra_label = "SSv2 extras (full)"
    else:
        extra_label = "SSv2 extras"
    plot_distribution(rows, output=args.output, title_suffix=suffix, extra_label=extra_label)
    pdf_path = args.pdf if args.pdf is not None else args.output.with_suffix(".pdf")
    if pdf_path.resolve() != args.output.resolve():
        plot_distribution(rows, output=pdf_path, title_suffix=suffix, extra_label=extra_label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
