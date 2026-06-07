#!/usr/bin/env python3
"""Plot last or best val/top1 vs MAE pretrain epoch (mean ± std over seeds).

Reads a W&B export CSV with columns ``{run} - val/top1`` and writes a PNG.

Example::

    uv run python scripts/plot_ft_mae_scaling_last_val_top1.py \\
        outputs/wandb_export_ft_mae_scaling_merged.csv --metric both
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

EPOCHS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
SEEDS = [42, 43, 44]
HEADS = ["meanpool", "perceiverQ16"]
HEAD_LABELS = {"meanpool": "Mean pool", "perceiverQ16": "Perceiver Q=16"}
HEAD_PALETTE = {"Mean pool": "#2563eb", "Perceiver Q=16": "#dc2626"}
LINE_ALPHA = 0.72
ERR_ALPHA = 0.28

METRIC_CONFIG = {
    "last": {
        "default_output": Path("outputs/ft_mae_scaling_last_val_top1_by_seed.png"),
        "ylabel": "Last validation top-1 accuracy",
        "title_line": "FT scaling: last val top-1 vs SSL depth",
    },
    "best": {
        "default_output": Path("outputs/ft_mae_scaling_best_val_top1_by_seed.png"),
        "ylabel": "Best validation top-1 accuracy",
        "title_line": "FT scaling: best val top-1 vs SSL depth",
    },
}


def parse_run_column(col: str) -> tuple[str, int, int] | None:
    """Return (head, pretrain_epoch, seed) from a W&B export column name."""
    m = re.match(r"^(.+?) - val/top1$", col)
    if not m:
        return None
    run = m.group(1)
    if run.startswith("meanpool-mae"):
        head, rest = "meanpool", run[len("meanpool-mae") :]
    elif run.startswith("perceiverQ16-mae"):
        head, rest = "perceiverQ16", run[len("perceiverQ16-mae") :]
    else:
        return None
    sm = re.match(r"^(\d+)-s(\d+)$", rest)
    if not sm:
        return None
    return head, int(sm.group(1)), int(sm.group(2))


def val_top1_per_run(df: pd.DataFrame, *, metric: str) -> dict[tuple[str, int, int], float]:
    """Map (head, pretrain_epoch, seed) -> last or max val/top1 in the series."""
    if metric not in ("last", "best"):
        raise ValueError(f"metric must be 'last' or 'best', got {metric!r}")
    out: dict[tuple[str, int, int], float] = {}
    for col in df.columns:
        parsed = parse_run_column(col)
        if parsed is None:
            continue
        head, ep, seed = parsed
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        value = float(series.iloc[-1] if metric == "last" else series.max())
        out[(head, ep, seed)] = value
    return out


def vals_to_frame(vals: dict[tuple[str, int, int], float]) -> pd.DataFrame:
    """Long-form table for seaborn: one row per (head, pretrain epoch, seed)."""
    rows: list[dict[str, object]] = []
    for (head, ep, seed), accuracy in vals.items():
        rows.append(
            {
                "pretrain_epoch": ep,
                "accuracy": accuracy,
                "head": HEAD_LABELS[head],
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def plot_scaling(
    vals: dict[tuple[str, int, int], float],
    *,
    output: Path,
    ylabel: str,
    title_line: str,
) -> None:
    """Draw mean ± std curves for both head types (seaborn)."""
    plot_df = vals_to_frame(vals)
    if plot_df.empty:
        raise ValueError("No val/top1 series found in the CSV.")

    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    sns.lineplot(
        data=plot_df,
        x="pretrain_epoch",
        y="accuracy",
        hue="head",
        hue_order=list(HEAD_PALETTE.keys()),
        palette=HEAD_PALETTE,
        estimator="mean",
        errorbar="sd",
        marker="o",
        markersize=7,
        linewidth=2.4,
        alpha=LINE_ALPHA,
        err_style="band",
        err_kws={"alpha": ERR_ALPHA, "edgecolor": "none"},
        ax=ax,
    )

    # Mark epochs with fewer than three seeds.
    counts = (
        plot_df.groupby(["head", "pretrain_epoch"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    means = plot_df.groupby(["head", "pretrain_epoch"], as_index=False)["accuracy"].mean()
    n_seeds = len(SEEDS)
    sparse = counts.merge(means, on=["head", "pretrain_epoch"]).query("n < @n_seeds")
    for row in sparse.itertuples(index=False):
        color = HEAD_PALETTE[row.head]
        ax.annotate(
            f"n={row.n}",
            (row.pretrain_epoch, row.accuracy),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
            color=color,
            alpha=0.9,
        )

    ax.set_xlabel("MAE pretrain epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{title_line}\n"
        "(mean ± std over seeds 42, 43, 44 when present in export)",
        pad=12,
    )
    ax.set_xticks(EPOCHS)
    ax.set_ylim(0.3, 0.6)
    ax.grid(True, alpha=0.35, linewidth=0.8)
    sns.despine(ax=ax, left=False, bottom=False)
    leg = ax.legend(loc="lower right", frameon=True, fancybox=True, framealpha=0.92)
    leg.get_frame().set_edgecolor("0.85")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="W&B metrics export CSV")
    parser.add_argument(
        "-m",
        "--metric",
        choices=["last", "best", "both"],
        default="both",
        help="Use last or best val/top1 per run (default: both PNGs)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG (only when --metric is last or best, not both)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    metrics = ["last", "best"] if args.metric == "both" else [args.metric]

    for metric in metrics:
        cfg = METRIC_CONFIG[metric]
        out = args.output if args.output is not None and len(metrics) == 1 else cfg["default_output"]
        vals = val_top1_per_run(df, metric=metric)
        plot_scaling(
            vals,
            output=out,
            ylabel=cfg["ylabel"],
            title_line=cfg["title_line"],
        )


if __name__ == "__main__":
    main()
