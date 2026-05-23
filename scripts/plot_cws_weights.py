#!/usr/bin/env python3
"""Plot T-CWS class-dependent ensemble weights as a single 3×C matrix heatmap."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from smth2smth.shared.data import parse_class_index

SEEDS = (42, 43, 44)
CLUSTER_VAL = Path("/Data/thomas.turkieh/smth2smth/data/val")


def _class_labels(n_classes: int = 33) -> list[str]:
    names = [f"c{i}" for i in range(n_classes)]
    if CLUSTER_VAL.is_dir():
        for d in sorted(CLUSTER_VAL.iterdir()):
            if not d.is_dir():
                continue
            idx = parse_class_index(d.name)
            if idx is not None and 0 <= idx < n_classes:
                short = re.sub(r"^\d+_", "", d.name)
                if len(short) > 22:
                    short = short[:19] + "..."
                names[idx] = short
    return names


def _load_cws_matrix(
    cache_dir: Path, exp_id: str, csv_path: Path | None
) -> np.ndarray:
    """Return (3, C) CWS weight matrix (rows = seeds, cols = classes)."""
    if csv_path is not None and csv_path.is_file():
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        mat = np.zeros((3, len(rows)), dtype=np.float64)
        for r in rows:
            c = int(r["class_idx"])
            mat[0, c] = float(r["w_s42"])
            mat[1, c] = float(r["w_s43"])
            mat[2, c] = float(r["w_s44"])
        return mat

    results_path = cache_dir / "ensemble_results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))
    row = next(r for r in data["holdout_metrics"] if r["exp"] == exp_id)
    return np.array(row["weights"], dtype=np.float64)


def plot_cws_matrix(
    mat: np.ndarray,
    *,
    class_names: list[str],
    exp_id: str,
    out_path: Path,
    annotate: bool = True,
) -> None:
    """Single heatmap: rows = seeds, columns = classes; each column sums to 1."""
    n_rows, n_cols = mat.shape
    fig_h = max(4.0, 0.55 * n_rows + 1.5)
    fig_w = max(14.0, 0.38 * n_cols + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_yticks(np.arange(n_rows), labels=[f"s{s}" for s in SEEDS[:n_rows]])
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(class_names, rotation=60, ha="right", fontsize=7)
    ax.set_xlabel("Class (true label index)")
    ax.set_ylabel("Ensemble member")
    ax.set_title(
        f"CWS coefficient matrix — {exp_id}\n"
        f"(each column: weights on simplex, Σᵢ wᵢ,c = 1)",
        fontsize=11,
    )

    if annotate:
        for i in range(n_rows):
            for j in range(n_cols):
                val = mat[i, j]
                if val < 0.04:
                    continue
                color = "white" if val > 0.55 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("weight w(seed, class)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "outputs/ensemble/videomaev2_3seed_v3_champion",
    )
    parser.add_argument("--exp", type=str, default="T-cws")
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "outputs/ensemble/analysis/T_ws/cws_T-ws_weights.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/ensemble/analysis/T_ws/cws_weights_matrix_T_cws.png",
    )
    parser.add_argument("--no-annotate", action="store_true")
    args = parser.parse_args()

    mat = _load_cws_matrix(args.cache_dir, args.exp, args.csv)
    names = _class_labels(mat.shape[1])
    out_path = args.output.resolve()
    plot_cws_matrix(
        mat,
        class_names=names,
        exp_id=args.exp,
        out_path=out_path,
        annotate=not args.no_annotate,
    )
    print(f"[plot] wrote {out_path}  shape={mat.shape}")


if __name__ == "__main__":
    main()
