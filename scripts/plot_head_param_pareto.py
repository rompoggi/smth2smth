#!/usr/bin/env python3
"""Param-Pareto plot: trainable head params (log x) vs best-val top-1 @ MAE500 (y)."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime, time, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
for p in (REPO / "src", SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from smth2smth.analysis.head_params import (
    count_head_params_for_spec,
    divspace_spec,
    meanpool_spec,
    perceiver_spec,
)

# Reuse unified-fleet loading / stab detection.
from plot_unified_fleet_scaling import (  # type: ignore[import-not-found]
    COLLECTED_LOGS,
    DIV_RE,
    MEANPOOL_RE,
    Q_RE,
    UNIFIED,
    VAL_COL,
    WANDB_SUMMARY,
    annotate_is_stab,
    build_recipe_index,
    load_rows,
)

DEFAULT_OUT = REPO / "outputs/unified_fleet/plots"
K_VALUES = [1, 3, 6, 9]
Q_VALUES = [2, 4, 8, 16, 32, 64]
TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\]")
STARTED_RE = re.compile(r"^# started: (\S+)", re.M)
EPOCH_ISO_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\][^\n]*Epoch (\d+)/50"
)
STEP_TS_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}|\d{2}:\d{2}:\d{2})\][^\n]*\bstep\b"
)
MIN_PLAUSIBLE_HOURS = 12.0
MAX_STEP_GAP_HOURS = 2.0
FLEET_HOSTS = [
    "ablette", "anchois", "anguille", "barbeau", "barbue", "baudroie", "brochet",
    "carrelet", "gardon", "gymnote", "labre", "lieu", "lotte", "mulet", "murene",
    "piranha", "raie", "requin", "rouget", "roussette", "saumon", "silure", "sole",
    "thon", "truite",
]


def aggregate_sem(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    out = (
        df.groupby(group_col)[VAL_COL]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    out["sem"] = out["std"] / np.sqrt(out["n"].clip(lower=1))
    return out.fillna({"sem": 0.0})


def _parse_group(run_name: str) -> tuple[str, str, int | None, int | None]:
    m = MEANPOOL_RE.match(run_name)
    if m:
        return "meanpool", "Mean pool", None, None
    m = Q_RE.match(run_name)
    if m:
        q = int(m.group(1))
        return "perceiver", f"Perceiver Q{q}", q, None
    m = DIV_RE.match(run_name)
    if m:
        k = int(m.group(1))
        return "divspace", f"DivST K{k}", None, k
    return "other", run_name, None, None


def head_params_for_group(family: str, q: int | None, k: int | None) -> int:
    if family == "meanpool":
        return count_head_params_for_spec(meanpool_spec())
    if family == "perceiver" and q is not None:
        return count_head_params_for_spec(perceiver_spec(q))
    if family == "divspace" and k is not None:
        return count_head_params_for_spec(divspace_spec(k))
    raise ValueError(f"unknown head family {family!r}")


def _to_naive(dt: datetime) -> datetime:
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


def _parse_step_timestamps(text: str, anchor: datetime) -> list[datetime]:
    """Collect absolute datetimes from ``[HH:MM:SS] step`` / ISO step lines."""
    cur_date = anchor.date()
    times: list[datetime] = []
    for m in STEP_TS_RE.finditer(text):
        ts = m.group(1)
        if "T" in ts:
            t = _to_naive(datetime.fromisoformat(ts))
        else:
            h, mi, s = (int(x) for x in ts.split(":"))
            t = datetime.combine(cur_date, time(h, mi, s))
            if times and t <= times[-1]:
                cur_date += timedelta(days=1)
                t = datetime.combine(cur_date, time(h, mi, s))
        times.append(t)
    return times


def _active_hours_from_steps(text: str, anchor: datetime) -> float | None:
    """Sum step-to-step gaps, skipping long idle periods (VM stop / resume)."""
    times = _parse_step_timestamps(text, anchor)
    if len(times) < 2:
        return None
    total = 0.0
    for prev, cur in zip(times, times[1:]):
        gap = (cur - prev).total_seconds() / 3600.0
        if 0.0 < gap <= MAX_STEP_GAP_HOURS:
            total += gap
    return total if total >= MIN_PLAUSIBLE_HOURS else None


def _wall_hours_epoch_iso(text: str, anchor: datetime) -> float | None:
    """Fallback: last ``# started`` to final ISO ``Epoch N/50`` timestamp."""
    epochs = EPOCH_ISO_RE.findall(text)
    if not epochs:
        return None
    end_ts, _ = max(epochs, key=lambda pair: int(pair[1]))
    end = _to_naive(datetime.fromisoformat(end_ts))
    if end <= anchor:
        return None
    return (end - anchor).total_seconds() / 3600.0


def parse_train_hours_from_text(text: str) -> float | None:
    """Active GPU hours from step logs; fallback to epoch ISO wall-clock."""
    if "Epoch 50/50" not in text:
        return None
    starts = STARTED_RE.findall(text)
    if not starts:
        return None
    anchor = _to_naive(datetime.fromisoformat(starts[-1]))
    active = _active_hours_from_steps(text, anchor)
    if active is not None:
        return active
    return _wall_hours_epoch_iso(text, anchor)


def parse_train_hours(log_path: Path) -> float | None:
    """Parse wall-clock training hours from a local log file."""
    if not log_path.is_file():
        p = REPO / log_path
        if not p.is_file():
            return None
        log_path = p
    return parse_train_hours_from_text(log_path.read_text(errors="ignore"))


def _candidate_log_paths(run_name: str, row: pd.Series) -> list[Path]:
    """All local log paths to try for one run (stale copies may be incomplete)."""
    seen: set[Path] = set()
    out: list[Path] = []
    for col in ("collected_log", "log_path_remote"):
        if col in row.index and pd.notna(row.get(col)):
            for p in (Path(str(row[col])), REPO / str(row[col])):
                if p not in seen:
                    seen.add(p)
                    out.append(p)
    collected_root = REPO / "outputs/unified_fleet/collected/logs"
    if collected_root.is_dir():
        for p in sorted(collected_root.rglob(f"{run_name}_*.log")):
            if p not in seen:
                seen.add(p)
                out.append(p)
    log_root = REPO / "logs/track_a"
    if log_root.is_dir():
        for p in sorted(log_root.rglob(f"{run_name}_*.log")):
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def parse_train_hours_remote(host: str, run_name: str) -> float | None:
    """Stream the full remote log when local copies are stale or truncated."""
    if not host or (isinstance(host, float) and pd.isna(host)):
        return None
    script = (
        f'LOG=$(ls -t /Data/romain.poggi/smth2smth/logs/track_a/*/{run_name}_*.log 2>/dev/null | head -1); '
        f'if [ -n "$LOG" ]; then cat "$LOG"; fi'
    )
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", str(host), script],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return parse_train_hours_from_text(r.stdout)


def resolve_train_hours(run_name: str, row: pd.Series) -> float | None:
    """Best wall-clock estimate: local logs, then SSH (primary host, then fleet scan)."""
    best: float | None = None
    for path in _candidate_log_paths(run_name, row):
        h = parse_train_hours(path)
        if h is not None and h >= MIN_PLAUSIBLE_HOURS and (best is None or h > best):
            best = h

    hosts_to_try: list[str] = []
    host = row.get("host") if "host" in row.index else None
    if host and not (isinstance(host, float) and pd.isna(host)):
        hosts_to_try.append(str(host).split(".")[0])
    for h in FLEET_HOSTS:
        if h not in hosts_to_try:
            hosts_to_try.append(h)

    if best is None or best < MIN_PLAUSIBLE_HOURS:
        for h in hosts_to_try:
            rh = parse_train_hours_remote(h, run_name)
            if rh is not None and rh >= MIN_PLAUSIBLE_HOURS and (best is None or rh > best):
                best = rh
    return best


def attach_train_hours(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["train_hours"] = [
        resolve_train_hours(str(row["run_name"]), row) if "run_name" in row.index else None
        for _, row in out.iterrows()
    ]
    return out


def _point_label(row: pd.Series) -> str:
    if row["family"] == "meanpool":
        return "MP"
    if row["family"] == "perceiver" and pd.notna(row.get("q")):
        return f"Q{int(row['q'])}"
    if row["family"] == "divspace" and pd.notna(row.get("k")):
        return f"K{int(row['k'])}"
    return str(row["label"])


def build_pareto_table(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[(df["ssl_ep"] == 500) & df["is_stab"]].copy()
    rows: list[dict] = []
    for run_name, grp in sub.groupby("run_name"):
        family, label, q, k = _parse_group(run_name)
        if family == "other":
            continue
        rows.append(
            {
                "run_name": run_name,
                "family": family,
                "label": label,
                "q": q,
                "k": k,
                "seed": int(grp["seed"].iloc[0]) if "seed" in grp else None,
                VAL_COL: float(grp[VAL_COL].iloc[0]),
                "head_params": head_params_for_group(family, q, k),
            }
        )
    if not rows:
        return pd.DataFrame()
    detail = pd.DataFrame(rows)
    merge_cols = [
        c for c in ("run_name", "collected_log", "log_path_remote", "host") if c in sub.columns
    ]
    detail = attach_train_hours(
        detail.merge(sub.drop_duplicates("run_name")[merge_cols], on="run_name", how="left")
    )
    agg = aggregate_sem(detail, "label")
    meta = detail.groupby("label").agg(
        family=("family", "first"),
        head_params=("head_params", "first"),
        train_hours_mean=("train_hours", "median"),
    )
    out = agg.merge(meta, on="label")
    out["q"] = out["label"].map(
        lambda s: int(m.group(1)) if (m := re.search(r"Q(\d+)", s)) else np.nan
    )
    out["k"] = out["label"].map(
        lambda s: int(m.group(1)) if (m := re.search(r"K(\d+)", s)) else np.nan
    )
    return out.sort_values("head_params")


def _pareto_mask(agg: pd.DataFrame) -> np.ndarray:
    """True for non-dominated points (maximize y, minimize x)."""
    xs = agg["head_params"].to_numpy()
    ys = agg["mean"].to_numpy()
    n = len(agg)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if xs[j] <= xs[i] and ys[j] >= ys[i] and (xs[j] < xs[i] or ys[j] > ys[i]):
                mask[i] = False
                break
    return mask


def plot_pareto(agg: pd.DataFrame, detail: pd.DataFrame, out_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    colors = {"meanpool": "#2563eb", "perceiver": "#0d9488", "divspace": "#7c2d12"}
    markers = {"meanpool": "o", "perceiver": "D", "divspace": "s"}
    legend_labels: set[str] = set()

    for family in ("meanpool", "perceiver", "divspace"):
        sub = agg[agg["family"] == family]
        if sub.empty:
            continue
        c, mk = colors[family], markers[family]
        legend_name = (
            family.replace("meanpool", "Mean pool")
            .replace("perceiver", "Perceiver")
            .replace("divspace", "DivSpaceTime")
        )
        ax.scatter(
            sub["head_params"],
            sub["mean"],
            marker=mk,
            s=90,
            c=c,
            edgecolors="white",
            linewidths=0.6,
            zorder=5,
            label=legend_name if legend_name not in legend_labels else None,
        )
        legend_labels.add(legend_name)
        for _, row in sub.iterrows():
            ax.annotate(
                _point_label(row),
                (row["head_params"], row["mean"]),
                textcoords="offset points",
                xytext=(6, 4),
                ha="left",
                fontsize=9,
                color=c,
                fontweight="bold",
            )

    frontier = agg[_pareto_mask(agg)].sort_values("head_params")
    if len(frontier) >= 2:
        ax.plot(
            frontier["head_params"],
            frontier["mean"],
            color="#111827",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Pareto frontier",
            zorder=2,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Trainable head parameters (log scale)")
    ax.set_ylabel("Best validation top-1 @ MAE500")
    ax.set_title(
        "Head capacity vs accuracy @ MAE500 SSL (stab)\n"
        "Mean best-val over finished seeds",
        fontsize=12,
    )
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    path = out_dir / "head_param_pareto_mae500.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def plot_accuracy_vs_wallclock(agg: pd.DataFrame, detail: pd.DataFrame, out_dir: Path) -> None:
    """Separate figure: mean best val top-1 vs wall-clock training time."""
    means = agg.dropna(subset=["train_hours_mean"]).copy()
    if means.empty:
        print("SKIP accuracy_vs_wallclock: no log timestamps found")
        return

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"meanpool": "#2563eb", "perceiver": "#0d9488", "divspace": "#7c2d12"}
    markers = {"meanpool": "o", "perceiver": "D", "divspace": "s"}
    legend_labels: set[str] = set()

    for family in ("meanpool", "perceiver", "divspace"):
        sub = means[means["family"] == family]
        if sub.empty:
            continue
        c, mk = colors[family], markers[family]
        legend_name = (
            family.replace("meanpool", "Mean pool")
            .replace("perceiver", "Perceiver")
            .replace("divspace", "DivSpaceTime")
        )
        ax.scatter(
            sub["train_hours_mean"],
            sub["mean"],
            marker=mk,
            s=90,
            c=c,
            edgecolors="white",
            linewidths=0.6,
            zorder=5,
            label=legend_name if legend_name not in legend_labels else None,
        )
        legend_labels.add(legend_name)
        for _, row in sub.iterrows():
            ax.annotate(
                _point_label(row),
                (row["train_hours_mean"], row["mean"]),
                textcoords="offset points",
                xytext=(6, 4),
                ha="left",
                fontsize=9,
                color=c,
                fontweight="bold",
            )

    ax.set_xlabel("Active training time (hours)")
    ax.set_ylabel("Best validation top-1 @ MAE500")
    ax.set_title(
        "Accuracy vs training cost @ MAE500 SSL (stab)\n"
        "Mean best-val; hours summed from step logs (idle/resume gaps excluded)",
        fontsize=12,
    )
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    path = out_dir / "head_param_train_hours_mae500.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unified", type=Path, default=UNIFIED)
    ap.add_argument("--wandb-summary", type=Path, default=WANDB_SUMMARY)
    ap.add_argument("--collected-logs", type=Path, default=COLLECTED_LOGS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_rows(args.unified, args.wandb_summary)
    df = annotate_is_stab(df, build_recipe_index(args.collected_logs))
    df = df[df["is_stab"]].copy()

    detail_rows = []
    sub = df[(df["ssl_ep"] == 500)].copy()
    for run_name, grp in sub.groupby("run_name"):
        family, label, q, k = _parse_group(run_name)
        if family == "other":
            continue
        detail_rows.append(
            {
                "run_name": run_name,
                "family": family,
                "label": label,
                "q": q,
                "k": k,
                VAL_COL: float(grp[VAL_COL].iloc[0]),
                "head_params": head_params_for_group(family, q, k),
            }
        )
    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        raise SystemExit("No MAE500 stab runs found for Pareto plot")

    merge_cols = [
        c for c in ("run_name", "collected_log", "log_path_remote", "host") if c in sub.columns
    ]
    detail = attach_train_hours(
        detail.merge(sub.drop_duplicates("run_name")[merge_cols], on="run_name", how="left")
    )
    agg = build_pareto_table(df)
    audit = args.out_dir / "head_param_pareto_inputs.csv"
    agg.to_csv(audit, index=False)
    detail.to_csv(args.out_dir / "head_param_pareto_runs.csv", index=False)
    print(f"Pareto inputs: {len(agg)} head groups -> {audit}")
    plot_pareto(agg, detail, args.out_dir)
    plot_accuracy_vs_wallclock(agg, detail, args.out_dir)


if __name__ == "__main__":
    main()
