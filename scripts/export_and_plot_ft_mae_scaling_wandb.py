#!/usr/bin/env python3
"""Export ft-mae-scaling W&B metrics and produce thesis scaling figures (Plots 1–5).

Excludes DivSpaceTimeK* runs. Plot1–4 use last-epoch val top-1 (``VAL_COL``); error bands ±1 SEM.

Outputs under outputs/ft_mae_scaling_thesis/:
  - wandb_ft_mae_scaling_metrics.csv (long + wide exports)
  - plot1_head_scaling_3seed.png
  - plot2_gap_meanpool_minus_q16.png
  - plot3_q_sweep_mae500_3seed.png
  - plot4_asymptote_bars.png
  - plot5_best_vs_last_val.png
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "outputs/ft_mae_scaling_thesis"
ENTITY = "romain-poggi-ecole-polytechnique"
PROJECT = "smth2smth-frame-ablation"
GROUP = "ft-mae-scaling"

SSL_EPOCHS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
SEEDS = [42, 43, 44]
Q_VALUES = [2, 4, 8, 16, 32, 64]

# Metric plotted on thesis figures (plot1–4, per-seed variants).
VAL_COL = "last_val_top1"
VAL_YLABEL = "Last-epoch validation top-1"

MEANPOOL_RE = re.compile(r"^meanpool-mae(\d+)-s(\d+)$")
Q16_RE = re.compile(r"^perceiverQ16-mae(\d+)-s(\d+)$")
Q_RE = re.compile(r"^perceiverQ(\d+)-mae(\d+)-s(\d+)$")

# In-flight runs omitted from plots (raw W&B CSV still includes them).
EXCLUDE_FROM_PLOTS: frozenset[str] = frozenset()

# Plot 1 SSL scaling: perceiverQ16-mae500-s42 is stab Q-sweep / trainonly (W&B replay), not the
# scaling-fleet run — do not mix with perceiverQ16-mae500-s43/s44 from track_a_diverse_arch2_perceiver.
PLOT1_SCALING_EXCLUDE: frozenset[str] = frozenset(
    {
        "perceiverQ16-mae500-s42",
    }
)

NOSTAB_SUFFIX = "-NoStab"


def strip_nostab(name: str) -> str:
    return name[: -len(NOSTAB_SUFFIX)] if name.endswith(NOSTAB_SUFFIX) else name


def run_is_stab(run_name: str) -> bool:
    """True if run used ``track_a_diverse_arch2_perceiver_stab`` (not SSL-scaling base Q16)."""
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


def _stab_legend_handles(perceiver_color: str = "#dc2626") -> list:
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=perceiver_color,
            markeredgecolor=perceiver_color,
            markersize=9,
            linestyle="None",
            label="Perceiver stab",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=perceiver_color,
            markeredgecolor=perceiver_color,
            markersize=9,
            linestyle="None",
            label="Perceiver base",
        ),
    ]


def _scatter_perceiver_stab_base(
    ax: plt.Axes,
    pts: pd.DataFrame,
    *,
    value_col: str = VAL_COL,
    color: str = "#dc2626",
    zorder: int = 5,
    size: int = 70,
) -> None:
    """Overlay ◆ stab / ○ base markers for perceiver rows (expects ``is_stab`` column)."""
    for is_stab, marker in ((True, "D"), (False, "o")):
        chunk = pts[pts["is_stab"] == is_stab]
        if chunk.empty:
            continue
        ax.scatter(
            chunk["ssl_pretrain_epoch"],
            chunk[value_col],
            marker=marker,
            s=size,
            c=color,
            edgecolors="white",
            linewidths=0.6,
            zorder=zorder,
        )


def _annotate_stab_mix(ax: plt.Axes, ep: int, y: float, n_stab: int, n_base: int, *, color: str) -> None:
    if n_stab == 0 and n_base == 0:
        return
    if n_base == 0:
        txt = "stab"
    elif n_stab == 0:
        txt = "base"
    else:
        txt = f"{n_stab} stab, {n_base} base"
    ax.annotate(
        txt,
        (ep, y),
        textcoords="offset points",
        xytext=(0, -14),
        ha="center",
        fontsize=8,
        color=color,
    )


def summary_for_plots(summary: pd.DataFrame) -> pd.DataFrame:
    """Drop in-flight runs that would distort seed aggregates."""
    out = summary[~summary["run_name"].isin(EXCLUDE_FROM_PLOTS)].copy()
    dropped = sorted(EXCLUDE_FROM_PLOTS & set(summary["run_name"]))
    if dropped:
        print(f"Excluded from plots (still training): {', '.join(dropped)}")
    return out


def saturating(ep: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """y = a - b * exp(-c * ep). Asymptote = a as ep -> inf."""
    return a - b * np.exp(-c * ep)


@dataclass(frozen=True)
class RunKey:
    run_name: str
    head: str  # meanpool | perceiverQ16 | perceiverQ4 | ...
    ssl_ep: int
    seed: int
    q: int | None  # for perceiver


def parse_run_name(name: str) -> RunKey | None:
    if "DivSpaceTime" in name:
        return None
    name = strip_nostab(name)
    m = MEANPOOL_RE.match(name)
    if m:
        ep, seed = int(m.group(1)), int(m.group(2))
        return RunKey(name, "meanpool", ep, seed, None)
    m = Q16_RE.match(name)
    if m:
        ep, seed = int(m.group(1)), int(m.group(2))
        return RunKey(name, "perceiverQ16", ep, seed, 16)
    m = Q_RE.match(name)
    if m:
        q, ep, seed = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return RunKey(name, f"perceiverQ{q}", ep, seed, q)
    return None


def fetch_all_runs() -> pd.DataFrame:
    import wandb

    api = wandb.Api()
    rows: list[dict] = []
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP}, per_page=200)
    for run in runs:
        cfg = run.config or {}
        name = cfg.get("training.wandb.name") or run.name
        key = parse_run_name(name)
        if key is None:
            continue
        try:
            hist = run.history(samples=100, keys=["epoch", "val/top1", "val/ema_top1", "_step"])
        except Exception as exc:
            print(f"WARN: no history for {name}: {exc}")
            continue
        if hist is None or hist.empty:
            continue
        hist = hist.dropna(subset=["epoch", "val/top1"], how="any")
        if hist.empty:
            continue
        hist["epoch"] = hist["epoch"].astype(int)
        for ep_ft, grp in hist.groupby("epoch"):
            val = float(grp["val/top1"].iloc[-1])
            rows.append(
                {
                    "run_name": name,
                    "head": key.head,
                    "ssl_pretrain_epoch": key.ssl_ep,
                    "seed": key.seed,
                    "q": key.q,
                    "ft_epoch": int(ep_ft),
                    "val_top1": val,
                    "wandb_state": run.state,
                    "wandb_id": run.id,
                }
            )
    df = pd.DataFrame(rows)
    return df


def epoch_curve_df(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per run: best and last val top-1."""
    ep = raw[raw["ft_epoch"] >= 0].copy()
    summary_rows = []
    for (name,), grp in ep.groupby(["run_name"]):
        meta = grp.iloc[0]
        by_ep = grp.groupby("ft_epoch")["val_top1"].last()
        best_val = float(by_ep.max())
        last_ep = int(by_ep.index.max())
        last_val = float(by_ep.loc[last_ep])
        finished_50 = last_ep >= 50
        summary_rows.append(
            {
                "run_name": name,
                "head": meta["head"],
                "ssl_pretrain_epoch": int(meta["ssl_pretrain_epoch"]),
                "seed": int(meta["seed"]),
                "q": meta["q"],
                "best_val_top1": best_val,
                "last_val_top1": last_val,
                "last_ft_epoch": last_ep,
                "finished_50": finished_50,
                "wandb_state": meta["wandb_state"],
            }
        )
    return pd.DataFrame(summary_rows)


def aggregate_sem(
    df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
) -> pd.DataFrame:
    out = (
        df.groupby(group_cols)[value_col]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    out["sem"] = out["std"] / np.sqrt(out["n"].clip(lower=1))
    out["sem"] = out["sem"].fillna(0.0)
    return out


def fit_saturation(ssl_eps: np.ndarray, y: np.ndarray) -> dict:
    """Fit a - b*exp(-c*ep); return params + R² + asymptote."""
    x = np.asarray(ssl_eps, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {"a": np.nan, "b": np.nan, "c": np.nan, "r2": np.nan, "asymptote": np.nan}
    p0 = (y.max(), y.max() - y.min(), 0.01)
    bounds = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    try:
        popt, _ = curve_fit(saturating, x, y, p0=p0, bounds=bounds, maxfev=20000)
    except Exception:
        return {"a": np.nan, "b": np.nan, "c": np.nan, "r2": np.nan, "asymptote": np.nan}
    a, b, c = popt
    yhat = saturating(x, a, b, c)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"a": a, "b": b, "c": c, "r2": r2, "asymptote": a}


def plot1_head_scaling(summary: pd.DataFrame, out_dir: Path, seed: int | None = None) -> dict:
    heads = ["meanpool", "perceiverQ16"]
    exclude = PLOT1_SCALING_EXCLUDE if seed is None else frozenset()
    sub = summary[
        (summary["head"].isin(heads))
        & (summary["ssl_pretrain_epoch"].isin(SSL_EPOCHS))
        & (~summary["run_name"].isin(exclude))
    ].copy()
    if seed is not None:
        sub = sub[sub["seed"] == seed]
    sub["is_stab"] = sub["run_name"].map(run_is_stab)

    if seed is None:
        curve = aggregate_sem(sub, VAL_COL, ["head", "ssl_pretrain_epoch"])
        y_col, err_col, n_col = "mean", "sem", "n"
    else:
        curve = sub.groupby(["head", "ssl_pretrain_epoch"], as_index=False).agg(
            mean=(VAL_COL, "mean"),
            n=(VAL_COL, "count"),
        )
        curve["sem"] = 0.0
        y_col, err_col, n_col = "mean", "sem", "n"

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.95)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"meanpool": "#2563eb", "perceiverQ16": "#dc2626"}
    labels = {"meanpool": "Mean pool", "perceiverQ16": "Perceiver Q=16"}
    fit_rows = []

    x_line = np.linspace(min(SSL_EPOCHS), max(SSL_EPOCHS), 200)
    legend_extra: list = []
    for head in heads:
        h = curve[curve["head"] == head].sort_values("ssl_pretrain_epoch")
        if h.empty:
            continue
        x = h["ssl_pretrain_epoch"].values
        y = h[y_col].values
        sem = h[err_col].values
        n = h[n_col].values
        pq = sub[sub["head"] == head] if head == "perceiverQ16" else None
        if seed is None:
            ax.errorbar(
                x,
                y,
                yerr=sem,
                fmt="o-" if head == "meanpool" else "-",
                color=colors[head],
                label=labels[head] if head == "meanpool" else None,
                capsize=4,
                linewidth=2.2,
                markersize=8 if head == "meanpool" else 0,
            )
            for xi, yi, ni in zip(x, y, n, strict=True):
                if ni < 3:
                    ax.annotate(
                        f"n={int(ni)}",
                        (xi, yi),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=9,
                        color=colors[head],
                    )
                if head == "perceiverQ16" and pq is not None:
                    g = pq[pq["ssl_pretrain_epoch"] == xi]
                    _annotate_stab_mix(
                        ax,
                        int(xi),
                        float(yi),
                        int(g["is_stab"].sum()),
                        int((~g["is_stab"]).sum()),
                        color=colors[head],
                    )
            if head == "perceiverQ16" and pq is not None:
                _scatter_perceiver_stab_base(ax, pq, color=colors[head], size=55, zorder=6)
                legend_extra = _stab_legend_handles(colors[head])
        else:
            if head == "meanpool":
                ax.plot(
                    x,
                    y,
                    "o-",
                    color=colors[head],
                    label=labels[head],
                    linewidth=2.2,
                    markersize=8,
                )
            else:
                ax.plot(x, y, "-", color=colors[head], linewidth=2.2, alpha=0.5)
                if pq is not None:
                    _scatter_perceiver_stab_base(ax, pq, color=colors[head])
                    for _, row in pq.iterrows():
                        tag = "stab" if row["is_stab"] else "base"
                        ax.annotate(
                            tag,
                            (row["ssl_pretrain_epoch"], row[VAL_COL]),
                            textcoords="offset points",
                            xytext=(0, -12),
                            ha="center",
                            fontsize=8,
                            color=colors[head],
                        )
                legend_extra = _stab_legend_handles(colors[head])
        fr = fit_saturation(x, y)
        fr["head"] = head
        fit_rows.append(fr)
        if np.isfinite(fr["a"]):
            ax.plot(
                x_line,
                saturating(x_line, fr["a"], fr["b"], fr["c"]),
                "--",
                color=colors[head],
                alpha=0.75,
                linewidth=1.8,
                label=f"{labels[head]} fit (→ {fr['asymptote']:.3f})",
            )

    ax.set_xlabel("SSL pretrain epoch")
    ax.set_ylabel(VAL_YLABEL)
    stab_note = "Perceiver: ◆ stab · ○ base (mean pool = official FT, not stab)"
    if seed is None:
        title = (
            "Head scaling at full parity (seeds 42–44)\n"
            "Mean ± 1 SEM; dashed: $a - b e^{-c\\cdot\\mathrm{ep}}$ fit · "
            + stab_note
        )
        path = out_dir / "plot1_head_scaling_3seed.png"
    else:
        title = (
            f"Head scaling at full parity (seed {seed} only)\n"
            f"Dashed: saturating fit · {stab_note}"
        )
        path = out_dir / f"plot1.{seed}.png"
    ax.set_title(title, fontsize=12)
    ax.set_xticks(SSL_EPOCHS)
    ax.set_ylim(0.35, 0.58)
    from matplotlib.lines import Line2D

    handles, labels = ax.get_legend_handles_labels()
    if legend_extra:
        handles = handles + [
            Line2D([0], [0], color="#dc2626", linestyle="-", linewidth=2, label="Perceiver Q=16"),
            *legend_extra,
        ]
        labels = labels + ["Perceiver Q=16", *[h.get_label() for h in legend_extra]]
    ax.legend(handles, labels, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")
    return {r["head"]: r for r in fit_rows}


def plot2_gap(summary: pd.DataFrame, out_dir: Path) -> None:
    heads = ["meanpool", "perceiverQ16"]
    sub = summary[
        (summary["head"].isin(heads))
        & (summary["ssl_pretrain_epoch"].isin(SSL_EPOCHS))
    ].copy()
    pivot = sub.pivot_table(
        index=["ssl_pretrain_epoch", "seed"],
        columns="head",
        values=VAL_COL,
    ).reset_index()
    # Paired seeds only (same seed must have both heads at this SSL epoch).
    pivot = pivot.dropna(subset=["meanpool", "perceiverQ16"], how="any")
    pivot["gap"] = pivot["meanpool"] - pivot["perceiverQ16"]
    agg = aggregate_sem(pivot, "gap", ["ssl_pretrain_epoch"])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = agg["ssl_pretrain_epoch"].values
    y = agg["mean"].values
    sem = agg["sem"].values
    ax.axhline(0, color="0.5", linewidth=1, linestyle=":")
    ax.errorbar(x, y, yerr=sem, fmt="o-", color="#7c3aed", capsize=4, linewidth=2.2, markersize=8)
    for xi, yi, ni in zip(x, y, agg["n"].values, strict=True):
        if ni < 3:
            ax.annotate(f"n={int(ni)}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("SSL pretrain epoch")
    ax.set_ylabel(f"Δ {VAL_YLABEL.lower()} (meanpool − Perceiver Q16)")
    ax.set_title(
        f"Head gap shrinks as backbone strengthens ({VAL_COL})\n"
        "Mean ± 1 SEM over seeds 42–44",
        fontsize=13,
    )
    ax.set_xticks(SSL_EPOCHS)
    fig.tight_layout()
    path = out_dir / "plot2_gap_meanpool_minus_q16.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def plot3_q_sweep(summary: pd.DataFrame, out_dir: Path, seed: int | None = None) -> None:
    sub = summary[
        (summary["ssl_pretrain_epoch"] == 500) & (summary["q"].notna())
    ].copy()
    mp = summary[(summary["head"] == "meanpool") & (summary["ssl_pretrain_epoch"] == 500)]
    if seed is not None:
        sub = sub[sub["seed"] == seed]
        mp = mp[mp["seed"] == seed]
    if sub.empty:
        print(f"SKIP plot3: no Q-sweep runs for seed={seed}")
        return

    sub["q_int"] = sub["q"].astype(int)
    sub["is_stab"] = sub["run_name"].map(run_is_stab)
    q_vals = sorted(sub["q_int"].unique())

    if seed is None:
        agg_q = aggregate_sem(sub, VAL_COL, ["q_int"])
        q_x, q_y, q_err = agg_q["q_int"], agg_q["mean"], agg_q["sem"]
        agg_mp = aggregate_sem(mp, VAL_COL, ["head"])
    else:
        agg_q = (
            sub.groupby("q_int", as_index=False)[VAL_COL]
            .mean()
            .rename(columns={VAL_COL: "mean"})
        )
        q_x, q_y, q_err = agg_q["q_int"], agg_q["mean"], np.zeros(len(agg_q))
        agg_mp = pd.DataFrame()

    perceiver_color = "#0d9488"
    fig, ax = plt.subplots(figsize=(9, 5.5))
    legend_extra: list = []
    if seed is None:
        ax.errorbar(
            q_x,
            q_y,
            yerr=q_err,
            fmt="-",
            color=perceiver_color,
            capsize=4,
            linewidth=2.2,
            markersize=0,
            label="Perceiver (by Q, mean)",
        )
        for qv in q_vals:
            g = sub[sub["q_int"] == qv]
            row = aggregate_sem(sub[sub["q_int"] == qv], VAL_COL, ["q_int"])
            if not row.empty and row.iloc[0]["n"] < 3:
                ax.annotate(
                    f"n={int(row.iloc[0]['n'])}",
                    (qv, row.iloc[0]["mean"]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=9,
                )
            if not g.empty:
                _annotate_stab_mix(
                    ax,
                    qv,
                    float(g[VAL_COL].mean()),
                    int(g["is_stab"].sum()),
                    int((~g["is_stab"]).sum()),
                    color=perceiver_color,
                )
        for is_stab, marker in ((True, "D"), (False, "o")):
            chunk = sub[sub["is_stab"] == is_stab]
            if chunk.empty:
                continue
            ax.scatter(
                chunk["q_int"],
                chunk[VAL_COL],
                marker=marker,
                s=65,
                c=perceiver_color,
                edgecolors="white",
                linewidths=0.6,
                zorder=5,
            )
        legend_extra = _stab_legend_handles(perceiver_color)
    else:
        ax.plot(
            q_x,
            q_y,
            "-",
            color=perceiver_color,
            linewidth=2.2,
            alpha=0.45,
            label="Perceiver (by Q)",
        )
        for is_stab, marker in ((True, "D"), (False, "o")):
            chunk = sub[sub["is_stab"] == is_stab]
            if chunk.empty:
                continue
            ax.scatter(
                chunk["q_int"],
                chunk[VAL_COL],
                marker=marker,
                s=80,
                c=perceiver_color,
                edgecolors="white",
                linewidths=0.6,
                zorder=5,
            )
        for _, row in sub.iterrows():
            tag = "stab" if row["is_stab"] else "base"
            ax.annotate(
                tag,
                (row["q_int"], row[VAL_COL]),
                textcoords="offset points",
                xytext=(0, -12),
                ha="center",
                fontsize=8,
                color=perceiver_color,
            )
        legend_extra = _stab_legend_handles(perceiver_color)

    if seed is None and not agg_mp.empty:
        m = agg_mp.iloc[0]
        ax.axhline(m["mean"], color="#2563eb", linestyle="--", linewidth=2, label="Mean pool (MAE500)")
        ax.fill_between(
            [min(q_vals) - 0.5, max(q_vals) + 0.5],
            m["mean"] - m["sem"],
            m["mean"] + m["sem"],
            color="#2563eb",
            alpha=0.15,
        )
    elif seed is not None and not mp.empty:
        m_val = float(mp[VAL_COL].iloc[0])
        ax.axhline(m_val, color="#2563eb", linestyle="--", linewidth=2, label="Mean pool (MAE500)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(q_vals)
    ax.set_xticklabels([str(q) for q in q_vals])
    ax.set_xlabel("Perceiver queries Q (log₂ scale)")
    ax.set_ylabel(VAL_YLABEL)
    stab_note = "Perceiver: ◆ stab · ○ base (mean pool = official FT)"
    if seed is None:
        title = (
            "Q sweep at MAE500 backbone (seeds 42–44)\n"
            f"Mean ± 1 SEM · {stab_note}"
        )
        path = out_dir / "plot3_q_sweep_mae500_3seed.png"
    else:
        title = f"Q sweep at MAE500 backbone (seed {seed} only)\n{stab_note}"
        path = out_dir / f"plot3.{seed}.png"
    ax.set_title(title, fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    if legend_extra:
        handles = handles + legend_extra
        labels = labels + [h.get_label() for h in legend_extra]
    ax.legend(handles, labels, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def plot4_asymptotes(fit_map: dict, summary: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Bar chart of fitted asymptotes with bootstrap CI from seed-level fits."""
    rows = []
    for head in ["meanpool", "perceiverQ16"]:
        sub = summary[(summary["head"] == head) & (summary["ssl_pretrain_epoch"].isin(SSL_EPOCHS))]
        seeds_present = sub["seed"].unique()
        asymptotes = []
        for seed in seeds_present:
            s = sub[sub["seed"] == seed]
            if len(s) < 3:
                continue
            fr = fit_saturation(s["ssl_pretrain_epoch"].values, s[VAL_COL].values)
            if np.isfinite(fr["asymptote"]):
                asymptotes.append(fr["asymptote"])
        if asymptotes:
            rows.append(
                {
                    "head": head,
                    "asymptote_mean": np.mean(asymptotes),
                    "asymptote_sem": np.std(asymptotes, ddof=1) / np.sqrt(len(asymptotes))
                    if len(asymptotes) > 1
                    else 0.0,
                    "n_seed_fits": len(asymptotes),
                    **{k: fit_map.get(head, {}).get(k, np.nan) for k in ("a", "b", "c", "r2")},
                }
            )

    tab = pd.DataFrame(rows)
    tab.to_csv(out_dir / "plot4_asymptote_fit_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels = ["Mean pool", "Perceiver Q=16"]
    xpos = np.arange(2)
    means = [
        tab.loc[tab["head"] == "meanpool", "asymptote_mean"].iloc[0]
        if (tab["head"] == "meanpool").any()
        else np.nan,
        tab.loc[tab["head"] == "perceiverQ16", "asymptote_mean"].iloc[0]
        if (tab["head"] == "perceiverQ16").any()
        else np.nan,
    ]
    sems = [
        tab.loc[tab["head"] == "meanpool", "asymptote_sem"].iloc[0]
        if (tab["head"] == "meanpool").any()
        else 0.0,
        tab.loc[tab["head"] == "perceiverQ16", "asymptote_sem"].iloc[0]
        if (tab["head"] == "perceiverQ16").any()
        else 0.0,
    ]
    colors = ["#2563eb", "#dc2626"]
    ax.bar(xpos, means, yerr=sems, capsize=6, color=colors, alpha=0.85, edgecolor="0.2")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fitted asymptote $a$ (val top-1)")
    ax.set_title("Same ceiling? Per-head $a - b e^{-c\\mathrm{ep}}$ asymptote\nMean of per-seed fits ± SEM", fontsize=11)
    for i, head in enumerate(["meanpool", "perceiverQ16"]):
        fr = fit_map.get(head, {})
        if fr:
            ax.text(
                i,
                means[i] + sems[i] + 0.005,
                f"$R^2$={fr.get('r2', float('nan')):.3f}",
                ha="center",
                fontsize=9,
            )
    fig.tight_layout()
    path = out_dir / "plot4_asymptote_bars.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")
    print(tab.to_string(index=False))
    return tab


def plot5_best_vs_last(summary: pd.DataFrame, out_dir: Path) -> None:
    fin = summary[summary["finished_50"]].copy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(fin["last_val_top1"], fin["best_val_top1"], alpha=0.55, s=40, c="0.35", edgecolors="none")
    lo = min(fin["last_val_top1"].min(), fin["best_val_top1"].min()) - 0.01
    hi = max(fin["last_val_top1"].max(), fin["best_val_top1"].max()) + 0.01
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="y = x")
    ax.set_xlabel("Last-epoch val top-1")
    ax.set_ylabel("Best val top-1 (max over FT)")
    ax.set_title(
        f"Best vs last val (finished ep50 only, n={len(fin)} runs)\n"
        "Tight diagonal → checkpoints had stabilized",
        fontsize=12,
    )
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    fig.tight_layout()
    path = out_dir / "plot5_best_vs_last_val.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-fetch", action="store_true", help="Use existing CSV only")
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_long = out_dir / "wandb_ft_mae_scaling_epoch_val_long.csv"
    csv_summary = out_dir / "wandb_ft_mae_scaling_run_summary.csv"

    if not args.skip_fetch:
        print("Fetching W&B history...")
        raw = fetch_all_runs()
        ep_rows = raw[raw["ft_epoch"] >= 0] if "ft_epoch" in raw.columns else raw
        ep_rows.to_csv(csv_long, index=False)
        summary = epoch_curve_df(raw)
        summary.to_csv(csv_summary, index=False)
        print(f"Wrote {csv_long} ({len(ep_rows)} rows)")
        print(f"Wrote {csv_summary} ({len(summary)} runs)")
    else:
        summary = pd.read_csv(csv_summary)

    plot_df = summary_for_plots(summary)
    plot_df.to_csv(out_dir / "wandb_ft_mae_scaling_run_summary_plots.csv", index=False)

    # Holes report uses plot subset only
    for head in ["meanpool", "perceiverQ16"]:
        for ep in SSL_EPOCHS:
            n = len(
                plot_df[(plot_df["head"] == head) & (plot_df["ssl_pretrain_epoch"] == ep)]
            )
            if n < 3:
                print(f"HOLE (plots): {head} mae{ep:03d} n={n}")

    fit_map = plot1_head_scaling(plot_df, out_dir)
    plot2_gap(plot_df, out_dir)
    plot3_q_sweep(plot_df, out_dir)
    for seed in SEEDS:
        plot1_head_scaling(plot_df, out_dir, seed=seed)
        plot3_q_sweep(plot_df, out_dir, seed=seed)
    plot4_asymptotes(fit_map, plot_df, out_dir)
    plot5_best_vs_last(plot_df, out_dir)

    # Combined fit table from plot1 (mean-curve fits)
    pd.DataFrame(
        [
            {"head": h, **{k: fit_map[h][k] for k in ("a", "b", "c", "r2", "asymptote")}}
            for h in fit_map
        ]
    ).to_csv(out_dir / "plot1_saturation_fit_on_seed_means.csv", index=False)

    # Wide CSV for external tools: plotted metric per (head, ssl_ep, seed)
    wide = plot_df.pivot_table(
        index=["head", "ssl_pretrain_epoch"],
        columns="seed",
        values=VAL_COL,
    )
    wide_path = out_dir / f"wandb_{VAL_COL}_wide.csv"
    wide.to_csv(wide_path)
    print(f"Wrote {wide_path}")


if __name__ == "__main__":
    main()
