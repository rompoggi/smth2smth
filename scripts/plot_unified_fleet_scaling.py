#!/usr/bin/env python3
"""Plot unified fleet scaling: SSL epoch sweep, Q@MAE500, K@MAE500 vs meanpool."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parents[1]
UNIFIED = REPO / "outputs/unified_fleet/unified_run_summary.csv"
WANDB_SUMMARY = REPO / "outputs/ft_mae_scaling_thesis/wandb_ft_mae_scaling_run_summary.csv"
COLLECTED_LOGS = REPO / "outputs/unified_fleet/collected/logs"
DEFAULT_OUT = REPO / "outputs/unified_fleet/plots"
HYDRA_RE = re.compile(r"^# hydra:.*experiment=(\S+)", re.M)
RECIPE_RE = re.compile(r"^# recipe: (\S+)", re.M)

VAL_COL = "best_val_top1"
LAST_VAL_COL = "last_val_top1"
YLABEL = "Best validation top-1"
SSL_EPOCHS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
Q_VALUES = [2, 4, 8, 16, 32, 64]
K_VALUES = [1, 3, 6, 9]

MEANPOOL_RE = re.compile(r"^meanpool-mae0*(\d+)-s(\d+)$")
Q_RE = re.compile(r"^perceiverQ(\d+)-mae0*(\d+)-s(\d+)$")
DIV_RE = re.compile(r"^DivSpaceTimeK(\d+)-mae0*(\d+)-s(\d+)$")
NOSTAB = "-NoStab"


def aggregate_sem(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = (
        df.groupby(group_cols)[VAL_COL]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    out["sem"] = out["std"] / np.sqrt(out["n"].clip(lower=1))
    return out.fillna({"sem": 0.0})


def _parse_ssl_ep(name: str) -> int | None:
    name = name.replace(NOSTAB, "")
    m = MEANPOOL_RE.match(name)
    if m:
        return int(m.group(1))
    m = Q_RE.match(name)
    if m:
        return int(m.group(2))
    m = DIV_RE.match(name)
    if m:
        return int(m.group(2))
    return None


def load_rows(unified: Path, wandb_summary: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if unified.is_file():
        u = pd.read_csv(unified)
        u["source"] = "unified"
        if "ssl_ep" not in u.columns:
            u["ssl_ep"] = u["run_name"].map(_parse_ssl_ep)
        frames.append(u)

    if wandb_summary.is_file():
        w = pd.read_csv(wandb_summary)
        w = w.rename(columns={"ssl_pretrain_epoch": "ssl_ep"})
        if VAL_COL not in w.columns and "best_val_top1" in w.columns:
            w[VAL_COL] = w["best_val_top1"]
        w["run_name"] = w["run_name"].str.replace(NOSTAB, "", regex=False)
        w["finished_50"] = w.get("finished_50", w.get("last_ft_epoch", 0) >= 50)
        w["source"] = "wandb"
        # meanpool + any perceiver/div rows not superseded by unified
        w = w[w["run_name"].str.match(r"^(meanpool|perceiverQ|DivSpaceTime)")]
        frames.append(w)

    if not frames:
        raise FileNotFoundError("No summary CSV found")

    df = pd.concat(frames, ignore_index=True)
    df["run_name"] = df["run_name"].astype(str).str.replace(NOSTAB, "", regex=False)
    df["ssl_ep"] = df["ssl_ep"].fillna(df["run_name"].map(_parse_ssl_ep))
    # Prefer unified (fleet logs/ckpts) over stale W&B export when both exist.
    _src_prio = {"wandb": 0, "unified": 1}
    df["_src_prio"] = df["source"].map(_src_prio).fillna(0)
    df = df.sort_values(["run_name", "_src_prio"])
    df = df.drop_duplicates(subset=["run_name"], keep="last").drop(columns=["_src_prio"])
    df = df[df["finished_50"].fillna(False)]
    if "best_val_top1" in df.columns:
        df[VAL_COL] = pd.to_numeric(df["best_val_top1"], errors="coerce")
    if LAST_VAL_COL not in df.columns:
        df[LAST_VAL_COL] = np.nan
    return df.dropna(subset=[VAL_COL])


def _experiment_is_stab(experiment: str) -> bool:
    return "_stab" in experiment or experiment.endswith("_perceiver_stab")


def _log_stab_from_path(log_path: object) -> bool | None:
    if log_path is None or (isinstance(log_path, float) and pd.isna(log_path)):
        return None
    p = Path(str(log_path))
    if not p.is_file():
        p = REPO / p
    if not p.is_file():
        return None
    head = p.read_text(errors="ignore")[:4000]
    recipe = RECIPE_RE.search(head)
    if recipe:
        return recipe.group(1) == "stab"
    m = HYDRA_RE.search(head)
    if not m:
        return None
    return _experiment_is_stab(m.group(1))


def build_recipe_index(collected_root: Path) -> dict[str, bool]:
    """Map run_name -> stab flag from collected log hydra headers."""
    idx: dict[str, bool] = {}
    if not collected_root.is_dir():
        return idx
    for log in collected_root.rglob("*.log"):
        head = log.read_text(errors="ignore")[:4000]
        recipe = RECIPE_RE.search(head)
        run = log.stem.rsplit("_", 1)[0]
        if recipe:
            idx[run] = recipe.group(1) == "stab"
            continue
        m = HYDRA_RE.search(head)
        if not m:
            continue
        idx[run] = _experiment_is_stab(m.group(1))
    return idx


def _wandb_perceiver_is_stab(run_name: str) -> bool:
    """Fallback for W&B-only rows without a collected log."""
    if run_name.endswith(NOSTAB):
        return False
    m = Q_RE.match(run_name)
    if not m:
        return False
    q, ep = int(m.group(1)), int(m.group(2))
    if ep != 500:
        return False
    if q == 16:
        return False
    return q in (2, 4, 8, 32, 64)


def annotate_is_stab(df: pd.DataFrame, recipe_idx: dict[str, bool]) -> pd.DataFrame:
    out = df.copy()

    def row_is_stab(row: pd.Series) -> bool:
        name = str(row["run_name"])
        if name.startswith("meanpool"):
            return True
        if name.startswith("DivSpaceTime"):
            return True
        for col in ("collected_log", "log_path_remote"):
            if col in row.index:
                from_log = _log_stab_from_path(row.get(col))
                if from_log is not None:
                    return from_log
        if name in recipe_idx:
            return recipe_idx[name]
        if row.get("wandb_recipe") == "stab":
            return True
        if row.get("source") == "unified" and name.startswith("perceiverQ8-"):
            return True
        if row.get("source") == "wandb" and name.startswith("perceiverQ"):
            return _wandb_perceiver_is_stab(name)
        return False

    out["is_stab"] = out.apply(row_is_stab, axis=1)
    return out


def saturating(ep: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """y = a - b * exp(-c * ep). Asymptote = a as ep -> inf."""
    return a - b * np.exp(-c * ep)


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


def _annotate_n(ax: plt.Axes, x, y, n: int, color: str) -> None:
    if n < 3:
        ax.annotate(
            f"n={n}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
            color=color,
        )


def plot_ssl_epoch_sweep(df: pd.DataFrame, out_dir: Path) -> None:
    heads = [
        ("meanpool", "Mean pool", "#2563eb", "o"),
        ("perceiverQ8", "Perceiver Q8", "#dc2626", "D"),
    ]
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    fig, ax = plt.subplots(figsize=(10, 6))
    x_line = np.linspace(min(SSL_EPOCHS), max(SSL_EPOCHS), 200)
    fit_rows: list[dict] = []

    for head_prefix, label, color, marker in heads:
        if head_prefix == "meanpool":
            sub = df[df["run_name"].str.match(r"^meanpool-mae")]
        else:
            q = int(head_prefix.replace("perceiverQ", ""))
            sub = df[df["run_name"].str.match(rf"^perceiverQ{q}-mae\d+-s\d+$")]
        sub = sub[sub["ssl_ep"].isin(SSL_EPOCHS)]
        sub = sub.dropna(subset=[VAL_COL])
        if sub.empty:
            continue
        agg = aggregate_sem(sub, ["ssl_ep"])
        for _, row in sub.iterrows():
            ax.scatter(
                row["ssl_ep"],
                row[VAL_COL],
                marker=marker,
                s=45,
                c=color,
                edgecolors="white",
                linewidths=0.4,
                alpha=0.95,
                zorder=5,
            )
        ax.errorbar(
            agg["ssl_ep"],
            agg["mean"],
            yerr=agg["sem"],
            fmt=f"{marker}-",
            color=color,
            capsize=4,
            linewidth=2.2,
            markersize=8,
            alpha=0.85,
            label=label,
            zorder=6,
        )
        for _, row in agg.iterrows():
            _annotate_n(ax, row["ssl_ep"], row["mean"], int(row["n"]), color)

        fr = fit_saturation(agg["ssl_ep"].values, agg["mean"].values)
        fr["head"] = head_prefix
        fit_rows.append(fr)
        if np.isfinite(fr["a"]):
            ax.plot(
                x_line,
                saturating(x_line, fr["a"], fr["b"], fr["c"]),
                "--",
                color=color,
                alpha=0.75,
                linewidth=1.8,
                label=f"{label} fit",
                zorder=4,
            )

    if fit_rows:
        pd.DataFrame(fit_rows).to_csv(out_dir / "ssl_epoch_saturation_fit.csv", index=False)

    ax.set_xlabel("SSL pretrain epoch (MAE backbone)")
    ax.set_ylabel(YLABEL)
    ax.set_title(
        "SSL epoch scaling: mean pool vs Perceiver Q8\n"
        "Mean ± 1 SEM (best val top-1); dashed: $a - b e^{-c\\cdot\\mathrm{ep}}$ fit",
        fontsize=12,
    )
    ax.set_xticks(SSL_EPOCHS)
    ax.legend(loc="center", fontsize=8, framealpha=0.92)

    coeff_lines: list[str] = []
    for head_prefix, label, _, _ in heads:
        fr = next((r for r in fit_rows if r["head"] == head_prefix), None)
        if fr is None or not np.isfinite(fr["a"]):
            continue
        coeff_lines.append(
            rf"{label}: $a={fr['a']:.3f},\ b={fr['b']:.3f},\ c={fr['c']:.4f}$"
            f" → {fr['asymptote']:.1%} ($R^2$={fr['r2']:.3f})"
        )
    if coeff_lines:
        ax.text(
            0.98,
            0.02,
            "\n".join(coeff_lines),
            transform=ax.transAxes,
            va="bottom",
            ha="right",
            fontsize=7,
            color="#334155",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#cbd5e1",
                "alpha": 0.95,
            },
        )

    fig.tight_layout()
    path = out_dir / "ssl_epoch_meanpool_q8_q16.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def plot_q_sweep_mae500(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[(df["ssl_ep"] == 500) & df["run_name"].str.match(r"^perceiverQ\d+-mae500-s\d+$")].copy()
    sub["q"] = sub["run_name"].map(lambda n: int(m.group(1)) if (m := Q_RE.match(n)) else np.nan)
    sub = sub[sub["q"].isin(Q_VALUES)]
    mp = df[(df["ssl_ep"] == 500) & df["run_name"].str.startswith("meanpool-mae")]
    if sub.empty:
        print("SKIP q_sweep: no perceiver Q@500")
        return

    agg_q = aggregate_sem(sub, ["q"])
    agg_mp = aggregate_sem(mp, ["ssl_ep"]) if not mp.empty else pd.DataFrame()

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    color = "#0d9488"
    ax.errorbar(
        agg_q["q"],
        agg_q["mean"],
        yerr=agg_q["sem"],
        fmt="-",
        color=color,
        capsize=4,
        linewidth=2.2,
        label="Perceiver (by Q)",
    )
    for _, row in sub.iterrows():
        ax.scatter(row["q"], row[VAL_COL], marker="D", s=55, c=color, edgecolors="white", linewidths=0.5, zorder=5)
    for _, row in agg_q.iterrows():
        _annotate_n(ax, row["q"], row["mean"], int(row["n"]), color)

    if not agg_mp.empty:
        m = agg_mp.iloc[0]
        ax.axhline(m["mean"], color="#2563eb", linestyle="--", linewidth=2, label="Mean pool MAE500")
        ax.fill_between(
            [min(Q_VALUES) - 0.5, max(Q_VALUES) + 0.5],
            m["mean"] - m["sem"],
            m["mean"] + m["sem"],
            color="#2563eb",
            alpha=0.15,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(Q_VALUES)
    ax.set_xticklabels([str(q) for q in Q_VALUES])
    ax.set_xlabel("Perceiver queries Q (log₂ scale)")
    ax.set_ylabel(YLABEL)
    ax.set_title(
        "Q sweep @ MAE500 backbone\nMean ± 1 SEM over finished seeds",
        fontsize=12,
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = out_dir / "q_sweep_mae500_vs_meanpool.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def plot_k_sweep_mae500(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[(df["ssl_ep"] == 500) & df["run_name"].str.match(r"^DivSpaceTimeK\d+-mae500-s\d+$")].copy()
    sub["k"] = sub["run_name"].map(lambda n: int(m.group(1)) if (m := DIV_RE.match(n)) else np.nan)
    sub = sub[sub["k"].isin(K_VALUES)]
    mp = df[(df["ssl_ep"] == 500) & df["run_name"].str.startswith("meanpool-mae")]
    if sub.empty:
        print("SKIP k_sweep: no DivSpaceTime @500")
        return

    agg_k = aggregate_sem(sub, ["k"])
    agg_mp = aggregate_sem(mp, ["ssl_ep"]) if not mp.empty else pd.DataFrame()

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    color = "#7c2d12"
    ax.errorbar(
        agg_k["k"],
        agg_k["mean"],
        yerr=agg_k["sem"],
        fmt="-",
        color=color,
        capsize=4,
        linewidth=2.2,
        label="DivSpaceTime (by K)",
    )
    for _, row in sub.iterrows():
        ax.scatter(row["k"], row[VAL_COL], marker="D", s=55, c=color, edgecolors="white", linewidths=0.5, zorder=5)
    for _, row in agg_k.iterrows():
        _annotate_n(ax, row["k"], row["mean"], int(row["n"]), color)

    if not agg_mp.empty:
        m = agg_mp.iloc[0]
        ax.axhline(m["mean"], color="#2563eb", linestyle="--", linewidth=2, label="Mean pool MAE500")
        ax.fill_between(
            [min(K_VALUES) - 0.5, max(K_VALUES) + 0.5],
            m["mean"] - m["sem"],
            m["mean"] + m["sem"],
            color="#2563eb",
            alpha=0.15,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel("DivSpaceTime K (log₂ scale)")
    ax.set_ylabel(YLABEL)
    ax.set_title(
        "K sweep @ MAE500 backbone (DivSpaceTime)\n"
        "Mean ± 1 SEM over finished seeds",
        fontsize=12,
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = out_dir / "k_sweep_mae500_vs_meanpool.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unified", type=Path, default=UNIFIED)
    ap.add_argument("--wandb-summary", type=Path, default=WANDB_SUMMARY)
    ap.add_argument("--collected-logs", type=Path, default=COLLECTED_LOGS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--include-non-stab",
        action="store_true",
        help="Include non-stab perceiver runs (default: stab recipe only).",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_rows(args.unified, args.wandb_summary)
    recipe_idx = build_recipe_index(args.collected_logs)
    df = annotate_is_stab(df, recipe_idx)
    if not args.include_non_stab:
        before = len(df)
        df = df[df["is_stab"]].copy()
        print(f"Stab filter: {before} -> {len(df)} finished runs")

    audit = args.out_dir / "plot_inputs_unified_scaling.csv"
    df.to_csv(audit, index=False)
    print(f"Plot inputs: {len(df)} finished runs -> {audit}")

    plot_ssl_epoch_sweep(df, args.out_dir)
    plot_q_sweep_mae500(df, args.out_dir)
    plot_k_sweep_mae500(df, args.out_dir)


if __name__ == "__main__":
    main()
