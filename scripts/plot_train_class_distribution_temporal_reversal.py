#!/usr/bin/env python3
"""Train label distribution under temporal reversal aug: stochastic and/or deterministic.

**Stochastic** (``--mode stochastic`` / ``both``, top): one dataset row per clip; each
load applies reversal+swap with probability ``p``. Stacked bar = ``n_k(1−p) + n_partner·p``.

**Deterministic** (``--mode deterministic`` / ``both``, bottom): each clip in a paired
class is duplicated — one row native label ``k``, one row partner label with reversed
frames. Stacked bar = ``n_k + n_partner`` (full exchange of mass each epoch).

Unpaired classes: single gray bar = ``n_k`` in both modes.

Run from repo root::

    PYTHONPATH=src uv run python scripts/plot_train_class_distribution_temporal_reversal.py
    PYTHONPATH=src uv run python scripts/plot_train_class_distribution_temporal_reversal.py \\
        --mode deterministic --output outputs/train_class_dist_deterministic.png
    PYTHONPATH=src uv run python scripts/plot_train_class_distribution_temporal_reversal.py \\
        --mode both --output outputs/train_class_dist_both_modes.png
"""

from __future__ import annotations

import argparse
import math
import sys
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smth2smth.shared.data import (
    TRACK_A_TEMPORAL_REVERSAL_PAIRS,
    build_track_a_temporal_reversal_map,
    collect_video_samples,
    parse_class_index,
)

_NUM_CLASS_SLOTS = 33


def _raw_counts_per_class(train_dir: Path) -> dict[int, int]:
    samples = collect_video_samples(train_dir)
    counts: Counter[int] = Counter()
    for _, label in samples:
        counts[int(label)] += 1
    return dict(counts)


def _folder_names_by_class_index(train_dir: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for p in sorted(train_dir.iterdir()):
        if not p.is_dir():
            continue
        idx = parse_class_index(p.name)
        if idx is not None:
            out[idx] = p.name
    return out


def _format_bar_annotation(value: float) -> str:
    if value <= 0:
        return ""
    if value >= 1000:
        return f"{value/1000:.1f}k" if value < 10000 else f"{value/1000:.0f}k"
    return f"{value:.0f}"


def _stochastic_displayed_components(
    raw: dict[int, int],
    *,
    prob: float,
    pair_map: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per class k: stay[k], incoming[k], disk n[k]."""
    n = np.array([float(raw.get(i, 0)) for i in range(_NUM_CLASS_SLOTS)])
    paired = set(pair_map.keys())
    stay = np.zeros(_NUM_CLASS_SLOTS)
    incoming = np.zeros(_NUM_CLASS_SLOTS)
    for k in range(_NUM_CLASS_SLOTS):
        nk = n[k]
        if k not in paired:
            stay[k] = nk
            incoming[k] = 0.0
        else:
            partner = int(pair_map[k])
            stay[k] = nk * (1.0 - prob)
            incoming[k] = float(n[partner]) * prob
    return stay, incoming, n


def _deterministic_displayed_components(
    raw: dict[int, int],
    *,
    pair_map: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Duplicate-entry aug: native n_k + n_partner mirrored rows with label k."""
    n = np.array([float(raw.get(i, 0)) for i in range(_NUM_CLASS_SLOTS)])
    paired = set(pair_map.keys())
    stay = np.zeros(_NUM_CLASS_SLOTS)
    incoming = np.zeros(_NUM_CLASS_SLOTS)
    for k in range(_NUM_CLASS_SLOTS):
        nk = n[k]
        if k not in paired:
            stay[k] = nk
            incoming[k] = 0.0
        else:
            partner = int(pair_map[k])
            stay[k] = nk
            incoming[k] = float(n[partner])
    return stay, incoming, n


def _deterministic_total_sample_rows(raw: dict[int, int], pair_map: dict[int, int]) -> int:
    """Sum of (native + mirror) list length = sum_k (stay[k]+incoming[k]) for deterministic."""
    s, inc, _ = _deterministic_displayed_components(raw, pair_map=pair_map)
    return int(np.sum(s + inc))


def _draw_stacked_panel(
    ax: plt.Axes,
    stay: np.ndarray,
    incoming: np.ndarray,
    n: np.ndarray,
    *,
    paired: set[int],
    subtitle: str,
) -> None:
    x = np.arange(_NUM_CLASS_SLOTS, dtype=float)
    total_displayed = stay + incoming

    for i in sorted(paired):
        ax.axvspan(i - 0.5, i + 0.5, color="#fff3d6", alpha=0.4, zorder=0)

    unpaired_mask = np.array([i not in paired for i in range(_NUM_CLASS_SLOTS)])
    paired_mask = ~unpaired_mask
    bar_w = 0.72

    if unpaired_mask.any():
        ax.bar(
            x[unpaired_mask],
            n[unpaired_mask],
            width=bar_w,
            color="#b0b8c4",
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )

    if paired_mask.any():
        xp = x[paired_mask]
        s = stay[paired_mask]
        inc = incoming[paired_mask]
        ax.bar(
            xp,
            s,
            width=bar_w,
            color="#3d6fa8",
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )
        ax.bar(
            xp,
            inc,
            width=bar_w,
            bottom=s,
            color="#e8943a",
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )
        for xi, tot in zip(xp, total_displayed[paired_mask], strict=True):
            if tot <= 0:
                continue
            ax.text(
                float(xi),
                float(tot) + max(n.max(), 1) * 0.012,
                _format_bar_annotation(float(tot)),
                ha="center",
                va="bottom",
                fontsize=7,
                color="#222222",
            )

    ax.set_xlabel("Class index (displayed label)")
    ax.set_ylabel("# training rows (expected or exact)")
    ax.set_title(subtitle, fontsize=10)
    ax.set_xticks(x[::2])
    ax.set_xticks(x, minor=True)
    ax.grid(axis="y", alpha=0.28)
    ax.set_xlim(-0.6, _NUM_CLASS_SLOTS - 0.4)


def _pair_footer_lines(folder_names: dict[int, str]) -> list[str]:
    lines: list[str] = []
    for a, b in TRACK_A_TEMPORAL_REVERSAL_PAIRS:
        na = folder_names.get(a, f"{a:03d}_?")
        nb = folder_names.get(b, f"{b:03d}_?")
        lines.append(f"• {na}  ↔  {nb}")
    return lines


def _save_figure(
    fig: plt.Figure,
    *,
    output: Path,
    prob: float,
    mode: str,
    folder_names: dict[int, str],
    train_dir: Path,
    deterministic_total_rows: int | None,
) -> None:
    pair_lines = _pair_footer_lines(folder_names)
    if mode == "stochastic":
        header_line = (
            f"Stochastic aug (p={prob:g}): stack = n_k(1−p) + n_partner·p per epoch "
            f"(one pass over disk clips)."
        )
    elif mode == "deterministic":
        header_line = (
            "Deterministic aug: each paired clip → 2 dataset rows (native + reversed partner label). "
            f"Stack = n_k + n_partner. Total training rows = {deterministic_total_rows}."
        )
    else:
        header_line = (
            f"Top: stochastic p={prob:g}. Bottom: deterministic duplicate rows. "
            f"Deterministic total rows = {deterministic_total_rows}."
        )
    wrapped_header = textwrap.fill(header_line, width=118)
    wrapped_pairs = "\n".join(textwrap.fill(line, width=118) for line in pair_lines)
    wrapped = wrapped_header + "\n\n" + wrapped_pairs

    handles = [
        Patch(facecolor="#b0b8c4", edgecolor="white", label="Unpaired"),
        Patch(facecolor="#3d6fa8", edgecolor="white", label="Native label k"),
        Patch(facecolor="#e8943a", edgecolor="white", label="From partner → k"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.10 if mode == "both" else 0.20),
        ncol=3,
        fontsize=8,
        frameon=True,
        fancybox=True,
    )

    fig.text(
        0.5,
        0.01 if mode == "both" else 0.015,
        wrapped,
        ha="center",
        va="bottom",
        fontsize=7.5,
        family="monospace",
        linespacing=1.15,
        transform=fig.transFigure,
    )

    bottom = 0.42 if mode == "both" else 0.38
    fig.subplots_adjust(left=0.07, right=0.99, top=0.94 if mode == "both" else 0.88, bottom=bottom)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=REPO_ROOT / "data" / "train",
        help="Split root with class subfolders (default: data/train).",
    )
    parser.add_argument(
        "--prob",
        type=float,
        default=0.5,
        help="Bernoulli p for stochastic mode (default: 0.5).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("stochastic", "deterministic", "both"),
        default="stochastic",
        help="Which distribution(s) to plot (default: stochastic).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "train_class_dist_temporal_reversal.png",
        help="Output PNG path.",
    )
    args = parser.parse_args()

    if not args.train_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.train_dir}")
    if not (0.0 <= args.prob <= 1.0) or math.isnan(args.prob):
        raise SystemExit("--prob must be in [0, 1]")

    raw = _raw_counts_per_class(args.train_dir)
    pair_map = build_track_a_temporal_reversal_map()
    paired: set[int] = set(pair_map.keys())
    folder_names = _folder_names_by_class_index(args.train_dir)
    det_total = _deterministic_total_sample_rows(raw, pair_map)

    if args.mode == "stochastic":
        stay, inc, n = _stochastic_displayed_components(raw, prob=args.prob, pair_map=pair_map)
        fig, ax = plt.subplots(figsize=(14, 6.2))
        _draw_stacked_panel(
            ax,
            stay,
            inc,
            n,
            paired=paired,
            subtitle=(
                f"Stochastic: E[rows with label k] = n_k(1−p) + n_partner·p  (p={args.prob:g})\n"
                f"{args.train_dir.resolve()}"
            ),
        )
        _save_figure(
            fig,
            output=args.output,
            prob=args.prob,
            mode="stochastic",
            folder_names=folder_names,
            train_dir=args.train_dir,
            deterministic_total_rows=None,
        )
    elif args.mode == "deterministic":
        stay, inc, n = _deterministic_displayed_components(raw, pair_map=pair_map)
        fig, ax = plt.subplots(figsize=(14, 6.2))
        _draw_stacked_panel(
            ax,
            stay,
            inc,
            n,
            paired=paired,
            subtitle=(
                f"Deterministic: each paired clip duplicated — rows with label k = n_k + n_partner\n"
                f"Total training rows = {det_total} (disk clips = {sum(raw.values())})\n"
                f"{args.train_dir.resolve()}"
            ),
        )
        _save_figure(
            fig,
            output=args.output,
            prob=args.prob,
            mode="deterministic",
            folder_names=folder_names,
            train_dir=args.train_dir,
            deterministic_total_rows=det_total,
        )
    else:
        s_st, i_st, n = _stochastic_displayed_components(raw, prob=args.prob, pair_map=pair_map)
        s_de, i_de, n2 = _deterministic_displayed_components(raw, pair_map=pair_map)
        fig, axes = plt.subplots(2, 1, figsize=(14, 10.5), sharex=True)
        _draw_stacked_panel(
            axes[0],
            s_st,
            i_st,
            n,
            paired=paired,
            subtitle=f"Stochastic (p={args.prob:g}): E[label k per epoch] = n_k(1−p) + n_partner·p",
        )
        _draw_stacked_panel(
            axes[1],
            s_de,
            i_de,
            n2,
            paired=paired,
            subtitle=(
                f"Deterministic: rows with label k = n_k + n_partner "
                f"(total rows {det_total} vs disk {sum(raw.values())})"
            ),
        )
        fig.suptitle(f"Train label mass: stochastic vs deterministic\n{args.train_dir.resolve()}", fontsize=11)
        _save_figure(
            fig,
            output=args.output,
            prob=args.prob,
            mode="both",
            folder_names=folder_names,
            train_dir=args.train_dir,
            deterministic_total_rows=det_total,
        )

    print(f"Wrote {args.output.resolve()}")
    print(f"Disk clips: {sum(raw.values())}")
    if args.mode in ("deterministic", "both"):
        print(f"Deterministic total training rows: {det_total}")


if __name__ == "__main__":
    main()
