#!/usr/bin/env python3
"""Plot last logged val/top1 vs Perceiver query count (diverse-heads train-only, ep500 SSL)."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
HEADS_CSV = REPO / "outputs/wandb_export_2026-05-30T23_35_15.815+02_00.csv"
MEANPOOL_CSV = REPO / "outputs/wandb_export_2026-05-30T23_05_31.361+02_00.csv"
OUT_PATH = REPO / "outputs/diverse_heads_q_sweep_val_top1_ep500.png"
MEANPOOL_RUN = "meanpool-mae500-s43"


def last_logged_val(series: pd.Series) -> float:
    """Last non-NaN val/top1 in a W&B export column (final training step)."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        raise ValueError("no logged val/top1 in column")
    return float(values.iloc[-1])


def parse_q_and_last(df: pd.DataFrame) -> tuple[list[tuple[int, str, float]], list[tuple[int, str, float]]]:
    """Return (attn_probe_rows, perceiver_rows) as (Q, run_name, last_val)."""
    attn: list[tuple[int, str, float]] = []
    perc: list[tuple[int, str, float]] = []

    for col in df.columns:
        if " - val/top1" not in col or col.endswith("__MIN") or col.endswith("__MAX"):
            continue
        run = col.replace(" - val/top1", "").strip()
        last_val = last_logged_val(df[col])

        if "attn-probe" in run or "attn_probe" in run:
            attn.append((1, run, last_val))
        elif m := re.search(r"perceiver-q(\d+)", run):
            perc.append((int(m.group(1)), run, last_val))

    attn.sort(key=lambda x: x[0])
    perc.sort(key=lambda x: x[0])
    return attn, perc


def meanpool_last_val(path: Path, run_name: str) -> float:
    df = pd.read_csv(path)
    col = f"{run_name} - val/top1"
    if col not in df.columns:
        raise KeyError(f"{col!r} not in {path}")
    return last_logged_val(df[col])


def main() -> None:
    df = pd.read_csv(HEADS_CSV)
    attn_rows, perc_rows = parse_q_and_last(df)
    control = meanpool_last_val(MEANPOOL_CSV, MEANPOOL_RUN)

    perc_q, perc_y = zip(*[(q, y) for q, _, y in perc_rows]) if perc_rows else ([], [])
    attn_q, attn_y = zip(*[(q, y) for q, _, y in attn_rows]) if attn_rows else ([], [])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)

    x_min, x_max = 0.5, 40
    ax.axhline(
        control,
        color="#2563eb",
        linestyle="--",
        linewidth=2.0,
        label=rf"Mean-pool control ({MEANPOOL_RUN}, last val top1)",
        zorder=1,
    )

    if perc_rows:
        ax.plot(
            perc_q,
            perc_y,
            "s-",
            color="#dc2626",
            linewidth=2.2,
            markersize=8,
            label="Perceiver (last val top1)",
            zorder=3,
        )
    if attn_rows:
        ax.scatter(
            attn_q,
            attn_y,
            s=90,
            color="#ea580c",
            edgecolors="#7c2d12",
            linewidths=1.2,
            marker="D",
            label="Attentive probe Q=1 (last val top1)",
            zorder=4,
        )

    all_q = list(attn_q) + list(perc_q)
    all_y = list(attn_y) + list(perc_y)
    for q, _run, y in attn_rows + perc_rows:
        color = "#9a3412" if q == 1 and attn_rows else "#991b1b"
        ax.annotate(
            f"{y:.1%}",
            (q, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            color=color,
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of queries Q (log2 scale)", fontsize=12)
    ax.set_ylabel("Official val top-1 accuracy (last logged step)", fontsize=12)
    ax.set_title(
        "Diverse heads Q-sweep (train-only, SSL ep500, seed 43)",
        fontsize=13,
        pad=12,
    )
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}" if v >= 1 else ""))
    ax.set_xlim(x_min, x_max)
    y_lo = min(min(all_y), control) * 0.98
    y_hi = max(max(all_y), control) * 1.02
    ax.set_ylim(y_lo, y_hi)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.grid(True, which="both", alpha=0.35)

    fig.text(
        0.5,
        0.01,
        f"Source: {HEADS_CSV.name} · control from {MEANPOOL_CSV.name} · all heads: ep500 SSL encoder",
        ha="center",
        fontsize=8,
        color="#64748b",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")
    print(f"Mean-pool control: {control:.4f} ({control:.1%})")
    print("Attn probe:", attn_rows)
    print("Perceiver:", perc_rows)


if __name__ == "__main__":
    main()
