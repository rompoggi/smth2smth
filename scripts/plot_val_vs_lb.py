#!/usr/bin/env python3
"""Val -> LB scatter for the MAE500 diverse-head ensembles.

x = held-out val top-1 (singles: full val; ensembles: 5-fold OOF).
y = Kaggle LB top-1 (public and private panels).
Singles faded; ensembles bold; y=x reference. Basic (no-TTA) inference on both
axes -> apples-to-apples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "outputs/ensemble/mae500_stab_val/val_lb_points.csv"
OUT = REPO / "outputs/ensemble/mae500_stab_val/plots/val_vs_lb_scatter.png"

SHORT = {
    "meanpool-mae500-s42": "meanpool",
    "perceiverQ8-mae500-s42": "Q8 s42",
    "perceiverQ8-mae500-s43": "Q8 s43",
    "perceiverQ8-mae500-s44": "Q8 s44",
    "DivSpaceTimeK9-mae500-s42": "DivST-K9",
    "perceiverQ16-mae500-s43": "Q16",
    "ens-seed-Q8x3": "seed (Q8x3)",
    "ens-arch-s42": "arch (mp+Q8+K9)",
    "ens-allaxes": "all-axes",
    "ens-diverse4-s42": "diverse-4",
}


def _panel(ax, df: pd.DataFrame, ycol: str, title: str) -> None:
    lo = min(df["val_top1"].min(), df[ycol].min()) - 0.6
    hi = max(df["val_top1"].max(), df[ycol].max()) + 0.6
    ax.plot([lo, hi], [lo, hi], ls="--", c="#9ca3af", lw=1, zorder=0, label="y = x")
    for _, r in df.iterrows():
        is_ens = r["kind"] == "ensemble"
        ax.scatter(
            r["val_top1"], r[ycol],
            s=190 if is_ens else 70,
            marker="*" if is_ens else "o",
            c="#dc2626" if is_ens else "#94a3b8",
            edgecolors="black" if is_ens else "none",
            linewidths=0.8, zorder=3 if is_ens else 2,
        )
        ax.annotate(
            SHORT.get(r["name"], r["name"]),
            (r["val_top1"], r[ycol]),
            textcoords="offset points", xytext=(7, 3),
            fontsize=8, fontweight="bold" if is_ens else "normal",
            color="#991b1b" if is_ens else "#475569",
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Val top-1 (%, basic OOF)")
    ax.set_ylabel(f"{title} LB top-1 (%)")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False, fontsize=8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=CSV)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.72)
    panels = [
        ("lb_public", "Public LB (basic)"),
        ("lb_private", "Private LB (basic)"),
        ("lb_public_tta", "Public LB (champion TTA)"),
        ("lb_private_tta", "Private LB (champion TTA)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, (col, title) in zip(axes.ravel(), panels):
        _panel(ax, df, col, title)
    fig.suptitle("Val -> LB  (MAE500 heads; x = basic val OOF)", y=0.997, fontsize=15)
    fig.text(0.5, 0.005, "grey o = single head     red * = ensemble (softmax)",
             ha="center", fontsize=10, color="#374151")
    fig.tight_layout(rect=(0, 0.02, 1, 0.99))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"wrote {args.out}")

    best_single = df[df.kind == "single"]
    best_ens = df[df.kind == "ensemble"]
    for col in ("lb_public", "lb_private", "lb_public_tta", "lb_private_tta"):
        bs = best_single[col].max()
        be = best_ens[col].max()
        print(f"{col}: best single={bs:.2f}  best ensemble={be:.2f}  gain={be-bs:+.2f}pp")


if __name__ == "__main__":
    main()
