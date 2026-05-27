#!/usr/bin/env python3
"""Render a tall heatmap of per-video classifier outputs on the test split.

Each row is one test clip; each column is one class. Values are usually row-wise
softmax probabilities so the strip reads as a confidence landscape (bright =
high mass on that class).

Typical shapes in this repo: ``(6913, 33)`` for the official test manifest and
``num_classes: 33`` (not full SSv2's 174 labels).

Examples::

    # From ensemble cache (after ``ensemble_track_a_videomae.py cache --also-test``)
    uv run python scripts/plot_submission_logits_heatmap.py \\
        --logits outputs/ensemble/videomaev2_3seed/logits_test_s42_champion.npy \\
        --title "Perceiver s42 — test logits (champion TTA)"

    # Recompute on GPU then plot
    uv run python scripts/plot_submission_logits_heatmap.py \\
        --checkpoint checkpoints/track_a/arch2-perceiver-q16-stab-s42.pt \\
        --test-dir data/test --tta champion
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from smth2smth.shared.data import parse_class_index

DEFAULT_OUT_DIR = REPO_ROOT / "report" / "figures" / "submission_logits_heatmap"


def load_logits_array(path: Path) -> np.ndarray:
    """Load a ``(N, C)`` float array from ``.npy``."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Logits file not found: {path}")
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D logits (N, C); got shape {arr.shape} from {path}")
    if arr.dtype.kind not in "fc":
        arr = arr.astype(np.float64)
    return np.ascontiguousarray(arr)


def scale_matrix(matrix: np.ndarray, mode: str) -> np.ndarray:
    """Apply per-row scaling for visualization."""
    if mode == "logits":
        return matrix.astype(np.float64, copy=False)
    if mode == "softmax":
        x = np.nan_to_num(matrix.astype(np.float64, copy=False), neginf=-1e9, posinf=1e9)
        x = x - x.max(axis=1, keepdims=True)
        exp = np.exp(x)
        return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)
    if mode == "zscore_row":
        x = matrix.astype(np.float64, copy=False)
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True)
        return (x - mu) / np.clip(sd, 1e-6, None)
    raise ValueError(f"Unknown scale mode: {mode!r}")


def parse_class_index_list(spec: str) -> list[int]:
    """Parse comma-separated class indices, e.g. ``'27'`` or ``'27,32'``."""
    spec = spec.strip().strip("'\"")
    if not spec or spec.lower() in ("none", "all"):
        return []
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def drop_class_columns(matrix: np.ndarray, exclude: list[int]) -> np.ndarray:
    """Remove columns whose class index is listed in ``exclude``."""
    if not exclude:
        return matrix
    n_cols = matrix.shape[1]
    keep = [c for c in range(n_cols) if c not in set(exclude)]
    if not keep:
        raise ValueError(f"exclude {exclude} would remove all {n_cols} columns")
    return matrix[:, keep]


def row_sort_order(matrix: np.ndarray, mode: str) -> np.ndarray:
    """Return indices that reorder rows for a clearer banded strip."""
    if mode == "none":
        return np.arange(matrix.shape[0])
    if mode in ("argmax", "reference"):
        pred = matrix.argmax(axis=1)
        conf = matrix.max(axis=1)
        # Stable sort: primary = predicted class, secondary = descending confidence.
        return np.lexsort((-conf, pred))
    raise ValueError(f"Unknown row sort mode: {mode!r}")


def column_sort_order(matrix: np.ndarray, mode: str) -> np.ndarray:
    """Return indices that reorder columns (classes)."""
    if mode == "index":
        return np.arange(matrix.shape[1])
    if mode == "variance":
        return np.argsort(-matrix.var(axis=0))
    raise ValueError(f"Unknown column sort mode: {mode!r}")


def subsample_rows(matrix: np.ndarray, stride: int) -> np.ndarray:
    """Keep every ``stride``-th row (``stride=1`` keeps all)."""
    stride = int(stride)
    if stride < 1:
        raise ValueError("row_stride must be >= 1")
    if stride == 1:
        return matrix
    return matrix[::stride]


def _val_label_root() -> Path | None:
    """First readable ``data/val`` tree (repo or cluster)."""
    for candidate in (REPO_ROOT / "data" / "val", Path("/Data/thomas.turkieh/smth2smth/data/val")):
        try:
            if candidate.is_dir():
                return candidate
        except OSError:
            continue
    return None


def class_labels(n_classes: int) -> list[str]:
    """Short human-readable labels when val folders are available."""
    names = [str(i) for i in range(n_classes)]
    val_root = _val_label_root()
    if val_root is None:
        return names
    for d in sorted(val_root.iterdir()):
        if not d.is_dir():
            continue
        idx = parse_class_index(d.name)
        if idx is not None and 0 <= idx < n_classes:
            short = re.sub(r"^\d+_", "", d.name)
            if len(short) > 18:
                short = short[:15] + "..."
            names[idx] = short
    return names


def render_logits_heatmap(
    matrix: np.ndarray,
    out_path: Path,
    *,
    title: str,
    scale_label: str,
    class_names: list[str],
    dpi: int,
    cmap: str,
    show_class_ticks: bool,
    vmin: float | None,
    vmax: float | None,
    ylabel: str,
    minimal: bool,
    square: bool,
    show_colorbar: bool,
    save_pdf: bool,
) -> None:
    """Write a single-panel heatmap PNG (and optional PDF)."""
    n_rows, n_cols = matrix.shape
    if square:
        side = max(3.0, 0.12 * max(n_rows, n_cols))
        fig, ax = plt.subplots(figsize=(side, side))
        aspect: str | float = "equal"
    else:
        fig_w = max(5.0, 0.22 * n_cols + 2.5)
        fig_h = min(28.0, max(6.0, n_rows / 280.0 + 2.0))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        aspect = "auto"

    im = ax.imshow(
        matrix,
        aspect=aspect,
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    if minimal:
        ax.set_axis_off()
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.subplots_adjust(0, 0, 1, 1)
    else:
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlabel("Class index")
        ax.set_ylabel(ylabel)
        if show_class_ticks and n_cols <= 40:
            ax.set_xticks(np.arange(n_cols))
            ax.set_xticklabels(class_names, rotation=70, ha="right", fontsize=6)
        else:
            ax.set_xticks([])
        ax.set_yticks([])
        if show_colorbar:
            cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            cbar.set_label(scale_label, fontsize=9)
        fig.tight_layout()

    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pad = 0 if minimal else 0.05
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=pad)
    if save_pdf and out_path.suffix.lower() == ".png":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=pad)
    plt.close(fig)


def collect_test_logits_from_checkpoint(
    checkpoint: Path,
    *,
    test_dir: Path,
    train_dir: Path,
    tta: str,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    """Run test inference and return ``(N, C)`` logits on CPU."""
    from smth2smth.ensemble.inference import TtaMode, collect_logits_for_videos
    from smth2smth.pipelines.submit import _resolve_test_videos

    _names, video_dirs = _resolve_test_videos(test_dir.resolve(), None)
    samples = [(p, 0) for p in video_dirs]
    tta_mode = TtaMode(tta)
    logits = collect_logits_for_videos(
        checkpoint.resolve(),
        samples,
        data_root=test_dir.resolve(),
        train_dir=train_dir.resolve(),
        tta_mode=tta_mode,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return logits.numpy()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--logits",
        type=Path,
        help="Path to (N, C) float .npy (e.g. logits_test_s42_champion.npy).",
    )
    src.add_argument(
        "--checkpoint",
        type=Path,
        help="Run test inference from this checkpoint, then plot (GPU).",
    )
    p.add_argument("--test-dir", type=Path, default=REPO_ROOT / "data" / "test")
    p.add_argument("--train-dir", type=Path, default=REPO_ROOT / "data" / "train")
    p.add_argument(
        "--tta",
        default="champion",
        choices=("none", "champion", "official_2x3"),
        help="TTA mode when using --checkpoint (default: champion).",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--save-logits",
        type=Path,
        default=None,
        help="When inferring, also write raw logits to this .npy path.",
    )
    p.add_argument(
        "--scale",
        default="softmax",
        choices=("softmax", "logits", "zscore_row"),
        help="Per-row transform before plotting (default: softmax).",
    )
    p.add_argument(
        "--sort-rows",
        default="argmax",
        choices=("argmax", "none", "reference"),
        help="Row order: argmax bands, manifest order, or order from --reference-logits.",
    )
    p.add_argument(
        "--reference-logits",
        type=Path,
        default=None,
        help="When --sort-rows=reference, .npy used to define row order (e.g. first model).",
    )
    p.add_argument(
        "--exclude-class-indices",
        default="27",
        help="Comma-separated class indices to drop (default: 27, never-trained / -inf).",
    )
    p.add_argument(
        "--sort-cols",
        default="index",
        choices=("index", "variance"),
        help="Column order: class index or descending variance.",
    )
    p.add_argument(
        "--row-stride",
        type=int,
        default=1,
        help="Plot every k-th row (preview / smaller file). Default 1 = all rows.",
    )
    p.add_argument(
        "--head-rows",
        type=int,
        default=None,
        help="Keep only the first N test clips (manifest order) before plotting.",
    )
    p.add_argument(
        "--minimal",
        action="store_true",
        help="No title, axes, colorbar, or border — colormap only.",
    )
    p.add_argument(
        "--square",
        action="store_true",
        help="Square figure with equal row/column aspect (e.g. 33x33 preview).",
    )
    p.add_argument(
        "--no-colorbar",
        action="store_true",
        help="Omit the colorbar (also implied by --minimal).",
    )
    p.add_argument(
        "--no-pdf",
        action="store_true",
        help="Do not write a sidecar PDF next to the PNG.",
    )
    p.add_argument("--title", default="", help="Figure title (default: derived from input path).")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"Output image (default: {DEFAULT_OUT_DIR}/<stem>.png).",
    )
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument(
        "--cmap",
        default="Blues",
        help="Matplotlib colormap (default: Blues, white=low / no black).",
    )
    p.add_argument(
        "--no-class-ticks",
        action="store_true",
        help="Hide x tick labels even when C is small.",
    )
    return p


def _class_names_after_exclude(n_cols: int, exclude: list[int]) -> list[str]:
    """Labels for columns kept after dropping ``exclude`` indices."""
    all_names = class_labels(n_cols)
    kept = [c for c in range(n_cols) if c not in set(exclude)]
    return [all_names[c] for c in kept]


def _prepare_matrix(
    raw: np.ndarray,
    *,
    scale: str,
    exclude: list[int],
    sort_rows: str,
    sort_cols: str,
    reference_raw: np.ndarray | None,
) -> tuple[np.ndarray, list[str], str]:
    """Scale, drop columns, sort; return matrix and class names for plotting."""
    n_cols = raw.shape[1]
    raw = drop_class_columns(raw, exclude)
    scaled = scale_matrix(raw, scale)
    if sort_rows == "reference":
        if reference_raw is None:
            raise ValueError("--sort-rows=reference requires --reference-logits")
        ref = drop_class_columns(reference_raw, exclude)
        if ref.shape[0] != scaled.shape[0]:
            raise ValueError(
                f"reference rows {ref.shape[0]} != logits rows {scaled.shape[0]}"
            )
        ref_scaled = scale_matrix(ref, scale)
        row_idx = row_sort_order(ref_scaled, "reference")
        ylabel = f"Test clip (order from reference, N={scaled.shape[0]})"
    else:
        row_idx = row_sort_order(scaled, sort_rows)
        ylabel = (
            f"Test clip (manifest order, N={scaled.shape[0]})"
            if sort_rows == "none"
            else f"Test clip (sorted, N={scaled.shape[0]})"
        )
    col_idx = column_sort_order(scaled, sort_cols)
    ordered = scaled[row_idx][:, col_idx]
    names = _class_names_after_exclude(n_cols, exclude)
    if sort_cols == "variance":
        names = [names[i] for i in col_idx]
    return ordered, names, ylabel


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    exclude = parse_class_index_list(str(args.exclude_class_indices))

    reference_raw: np.ndarray | None = None
    if args.sort_rows == "reference":
        if args.reference_logits is None:
            raise SystemExit("--sort-rows=reference requires --reference-logits")
        reference_raw = load_logits_array(args.reference_logits)

    if args.logits is not None:
        raw = load_logits_array(args.logits)
        stem = args.logits.stem
        default_title = stem.replace("_", " ")
    else:
        assert args.checkpoint is not None
        raw = collect_test_logits_from_checkpoint(
            args.checkpoint,
            test_dir=args.test_dir,
            train_dir=args.train_dir,
            tta=args.tta,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        if args.save_logits is not None:
            out_npy = args.save_logits.resolve()
            out_npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_npy, raw)
            print(f"[save] logits {raw.shape} -> {out_npy}")
        stem = args.checkpoint.stem
        default_title = f"{stem} — test ({args.tta} TTA)"

    if args.head_rows is not None:
        n = int(args.head_rows)
        if n < 1:
            raise SystemExit("--head-rows must be >= 1")
        raw = raw[:n]
        if reference_raw is not None:
            reference_raw = reference_raw[:n]

    ordered, names, ylabel = _prepare_matrix(
        raw,
        scale=args.scale,
        exclude=exclude,
        sort_rows=args.sort_rows,
        sort_cols=args.sort_cols,
        reference_raw=reference_raw,
    )
    plot_mat = subsample_rows(ordered, args.row_stride)

    scale_labels = {
        "softmax": "P(class | video)",
        "logits": "logit",
        "zscore_row": "row z-score",
    }
    title = args.title or default_title
    if args.row_stride > 1:
        title = f"{title} (every {args.row_stride} rows)"

    out = args.output
    if out is None:
        out = DEFAULT_OUT_DIR / f"{stem}_heatmap.png"
    out = Path(out)
    if out.suffix == "":
        out = out.with_suffix(".png")

    vmin: float | None = 0.0 if args.scale == "softmax" else None
    vmax: float | None = 1.0 if args.scale == "softmax" else None

    minimal = bool(args.minimal)
    render_logits_heatmap(
        plot_mat,
        out,
        title=title,
        scale_label=scale_labels[args.scale],
        class_names=names,
        dpi=args.dpi,
        cmap=args.cmap,
        show_class_ticks=not args.no_class_ticks and not minimal,
        vmin=vmin,
        vmax=vmax,
        ylabel=ylabel,
        minimal=minimal,
        square=bool(args.square),
        show_colorbar=not args.no_colorbar and not minimal,
        save_pdf=not args.no_pdf and not minimal,
    )
    print(f"[plot] {raw.shape} -> scaled {plot_mat.shape} -> {out.resolve()}")
    pdf = out.with_suffix(".pdf")
    if pdf.is_file():
        print(f"[plot] also wrote {pdf.resolve()}")


if __name__ == "__main__":
    main()
