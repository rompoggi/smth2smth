#!/usr/bin/env python3
"""Plot SSL pretrain epoch vs val/top1 from a W&B multi-run CSV export."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parents[1]
CSV_PATH = REPO / "outputs/wandb_export_2026-05-30T23_05_31.361+02_00.csv"
OUT_PATH = REPO / "outputs/ft_mae_scaling_s43_val_top1_vs_pretrain_epochs.png"


def last_logged_val(series: pd.Series) -> float:
    """Last non-NaN val/top1 in a W&B export column (final training step)."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        raise ValueError("no logged val/top1 in column")
    return float(values.iloc[-1])


def saturation(ep: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """a − b·exp(−c·ep): asymptote ``a`` as pretrain epochs grow."""
    return a - b * np.exp(-c * ep)


def pretrain_epochs(run_name: str) -> int:
    m = re.search(r"mae(\d+)", run_name)
    if not m:
        raise ValueError(f"cannot parse pretrain epochs from {run_name!r}")
    return int(m.group(1))


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    mean_rows: list[tuple[int, float]] = []
    perc_rows: list[tuple[int, float]] = []

    for col in df.columns:
        if " - val/top1" not in col or col.endswith("__MIN") or col.endswith("__MAX"):
            continue
        run = col.replace(" - val/top1", "").strip()
        ep = pretrain_epochs(run)
        last_val = last_logged_val(df[col])

        if run.startswith("meanpool-"):
            mean_rows.append((ep, last_val))
        elif run.startswith("perceiverQ16-"):
            perc_rows.append((ep, last_val))

    mean_rows.sort(key=lambda x: x[0])
    perc_rows.sort(key=lambda x: x[0])

    mean_x, mean_y = zip(*mean_rows)
    perc_x, perc_y = zip(*perc_rows)
    mean_x_arr = np.asarray(mean_x, dtype=float)
    mean_y_arr = np.asarray(mean_y, dtype=float)

    y_max = float(mean_y_arr.max())
    popt, _ = curve_fit(
        saturation,
        mean_x_arr,
        mean_y_arr,
        p0=[y_max + 0.005, y_max - float(mean_y_arr.min()), 0.01],
        bounds=([y_max, 0.0, 1e-6], [1.0, 1.0, 1.0]),
        maxfev=20_000,
    )
    a_fit, b_fit, c_fit = popt
    y_hat = saturation(mean_x_arr, *popt)
    ss_res = float(np.sum((mean_y_arr - y_hat) ** 2))
    ss_tot = float(np.sum((mean_y_arr - mean_y_arr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    ep_min = float(mean_x_arr.min())
    ep_curve = np.linspace(ep_min, 550, 300)
    y_curve = saturation(ep_curve, *popt)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)

    ax.plot(
        ep_curve,
        y_curve,
        "-",
        color="#1d4ed8",
        linewidth=1.8,
        alpha=0.55,
        label=rf"Mean-pool fit: $a - b e^{{-c\cdot ep}}$  ($R^2$={r2:.3f})",
        zorder=2,
    )
    ax.plot(
        mean_x,
        mean_y,
        "o-",
        color="#2563eb",
        linewidth=2.2,
        markersize=7,
        label="Mean-pool (last val top1)",
        zorder=3,
    )
    ax.plot(
        perc_x,
        perc_y,
        "s-",
        color="#dc2626",
        linewidth=2.2,
        markersize=7,
        label="Perceiver Q16 (last val top1)",
        zorder=3,
    )

    for x, y in mean_rows:
        ax.annotate(f"{y:.1%}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7, color="#1e40af")
    for x, y in perc_rows:
        ax.annotate(f"{y:.1%}", (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=7, color="#991b1b")

    ax.set_xlabel("SSL pretrain epochs", fontsize=12)
    ax.set_ylabel("Official val top-1 accuracy (last logged step)", fontsize=12)
    ax.set_title("FT scaling (seed 43, T=4) — honest val / W&B export", fontsize=13, pad=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xticks(sorted({*mean_x, *perc_x}))
    ax.set_ylim(0.0, max(max(mean_y), max(perc_y)) * 1.08)
    fit_text = (
        rf"$a={a_fit:.3f},\ b={b_fit:.3f},\ c={c_fit:.4f}$"
        f"\nasymptote ≈ {a_fit:.1%}"
    )
    ax.text(
        0.02,
        0.98,
        fit_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        color="#1e3a8a",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#93c5fd", "alpha": 0.9},
    )
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.35)

    fig.text(
        0.5,
        0.01,
        "Source: outputs/wandb_export_2026-05-30T23_05_31.361+02_00.csv · group ft-mae-scaling · seed 43",
        ha="center",
        fontsize=8,
        color="#64748b",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")
    print("Mean-pool last:", dict(mean_rows))
    print("Perceiver last:", dict(perc_rows))
    print(f"Saturation fit: a={a_fit:.6f}, b={b_fit:.6f}, c={c_fit:.6f}, R²={r2:.4f}")


if __name__ == "__main__":
    main()
