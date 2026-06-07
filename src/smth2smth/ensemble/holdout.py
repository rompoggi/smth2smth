"""Official-val stratified holdout split (matches ``train.py``)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from smth2smth.shared.data import collect_video_samples
from smth2smth.shared.utils.splits import VideoSample, label_counts, split_train_val_stratified

MANIFEST_VERSION = 1


def sample_key(video_dir: Path) -> str:
    """Stable string id for a clip (folder name under val)."""
    return video_dir.name


def build_official_val_holdout(
    val_dir: Path,
    *,
    holdout_ratio: float = 0.1,
    split_seed: int = 42,
) -> list[VideoSample]:
    """Return the stratified holdout val clips used for Mix fitting.

    Uses the same ``split_train_val_stratified`` call as ``train.py`` when
    ``official_val_holdout_ratio > 0``. The split RNG seed is ``split_seed``
    (typically 42), **not** the per-run training seed.

    Args:
        val_dir: Official validation root (``data/val``).
        holdout_ratio: Fraction held out per class (default 0.1).
        split_seed: RNG seed for the stratified split.

    Returns:
        List of ``(video_dir, label)`` holdout samples.
    """
    val_dir = val_dir.resolve()
    all_val = collect_video_samples(val_dir)
    _train_part, holdout = split_train_val_stratified(
        all_val, val_ratio=holdout_ratio, seed=split_seed
    )
    return holdout


def write_holdout_manifest(
    holdout: list[VideoSample],
    path: Path,
    *,
    meta: dict[str, Any] | None = None,
) -> None:
    """Persist holdout clip ids and labels for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "keys": [sample_key(vd) for vd, _ in holdout],
        "labels": [int(lab) for _, lab in holdout],
        "paths": [str(vd.resolve()) for vd, _ in holdout],
    }
    if meta:
        payload.update(meta)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_holdout_manifest(path: Path) -> dict[str, Any]:
    """Load a holdout manifest written by :func:`write_holdout_manifest`."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("version", 0)) != MANIFEST_VERSION:
        raise ValueError(f"Unsupported holdout manifest version {data.get('version')!r} in {path}")
    return data


def read_holdout_keys(path: Path) -> frozenset[str]:
    """Return the set of clip folder names in a holdout manifest."""
    data = read_holdout_manifest(path)
    return frozenset(str(k) for k in data["keys"])


def apply_holdout_manifest(
    val_samples: list[VideoSample],
    manifest_path: Path,
) -> tuple[list[VideoSample], list[VideoSample]]:
    """Split official val clips into (train-part, holdout) using a frozen manifest.

    Args:
        val_samples: All clips under ``data/val``.
        manifest_path: JSON with ``keys`` listing holdout clip folder names.

    Returns:
        ``(val_for_train, holdout)`` — clips not in the manifest vs manifest clips.

    Raises:
        ValueError: If manifest keys are missing from ``val_samples`` or unknown keys
            appear in the manifest.
    """
    manifest_path = manifest_path.resolve()
    holdout_keys = read_holdout_keys(manifest_path)
    by_key: dict[str, VideoSample] = {sample_key(vd): (vd, lab) for vd, lab in val_samples}

    missing = sorted(holdout_keys - by_key.keys())
    if missing:
        raise ValueError(
            f"holdout manifest {manifest_path} lists {len(missing)} keys not in val_dir "
            f"(e.g. {missing[:3]})"
        )

    holdout: list[VideoSample] = []
    for key in sorted(holdout_keys):
        holdout.append(by_key[key])

    val_for_train = [(vd, lab) for vd, lab in val_samples if sample_key(vd) not in holdout_keys]
    extra = holdout_keys - {sample_key(vd) for vd, _ in holdout}
    if extra:
        raise ValueError(f"internal error: unassigned holdout keys {extra}")

    return val_for_train, holdout


def class_distribution_report(
    train_samples: list[VideoSample],
    val_samples: list[VideoSample],
    holdout_samples: list[VideoSample],
    *,
    holdout_ratio: float,
) -> dict[str, Any]:
    """Summarise per-class counts and how representative the holdout is.

    Args:
        train_samples: Official train clips.
        val_samples: Full official val clips.
        holdout_samples: Frozen holdout subset of val.
        holdout_ratio: Target fraction per class (for diagnostics).

    Returns:
        JSON-serialisable report dict.
    """
    train_c = label_counts(train_samples)
    val_c = label_counts(val_samples)
    hold_c = label_counts(holdout_samples)
    val_train_c = label_counts(
        [
            (vd, lab)
            for vd, lab in val_samples
            if sample_key(vd) not in {sample_key(h) for h, _ in holdout_samples}
        ]
    )

    all_labels = sorted(set(train_c) | set(val_c))
    n_train = sum(train_c.values())
    n_val = sum(val_c.values())
    n_hold = sum(hold_c.values())

    per_class: list[dict[str, Any]] = []
    max_holdout_frac_err = 0.0
    max_train_vs_val_prop_err = 0.0

    for lab in all_labels:
        nt = train_c.get(lab, 0)
        nv = val_c.get(lab, 0)
        nh = hold_c.get(lab, 0)
        nvt = val_train_c.get(lab, 0)
        p_train = nt / n_train if n_train else 0.0
        p_val = nv / n_val if n_val else 0.0
        hold_frac = nh / nv if nv else 0.0
        holdout_frac_err = abs(hold_frac - holdout_ratio) if nv else 0.0
        max_holdout_frac_err = max(max_holdout_frac_err, holdout_frac_err)
        max_train_vs_val_prop_err = max(max_train_vs_val_prop_err, abs(p_train - p_val))
        per_class.append(
            {
                "label": lab,
                "train": nt,
                "val_total": nv,
                "val_holdout": nh,
                "val_for_train": nvt,
                "holdout_fraction": round(hold_frac, 4),
                "p_train": round(p_train, 6),
                "p_val": round(p_val, 6),
            }
        )

    return {
        "n_train": n_train,
        "n_val_total": n_val,
        "n_holdout": n_hold,
        "n_val_for_train": n_val - n_hold,
        "n_classes_train": len(train_c),
        "n_classes_val": len(val_c),
        "holdout_ratio_target": holdout_ratio,
        "max_per_class_holdout_fraction_error": round(max_holdout_frac_err, 4),
        "max_abs_train_vs_val_proportion_error": round(max_train_vs_val_prop_err, 6),
        "per_class": per_class,
    }


def build_and_write_holdout_clean(
    *,
    train_dir: Path,
    val_dir: Path,
    output_path: Path,
    holdout_ratio: float = 0.15,
    split_seed: int = 42,
) -> dict[str, Any]:
    """Build seed-42 stratified holdout, verify distributions, write manifest.

    Args:
        train_dir: ``data/train``.
        val_dir: ``data/val``.
        output_path: Destination JSON (e.g. ``data/holdout_clean.json``).
        holdout_ratio: Per-class holdout fraction on val (default 15%).
        split_seed: RNG seed (default 42).

    Returns:
        The distribution report dict (also embedded in the manifest).
    """
    train_dir = train_dir.resolve()
    val_dir = val_dir.resolve()
    train_samples = collect_video_samples(train_dir)
    val_samples = collect_video_samples(val_dir)
    val_for_train, holdout = split_train_val_stratified(
        val_samples, val_ratio=holdout_ratio, seed=split_seed
    )

    # Round-trip check: manifest reproduces the same holdout set.
    output_path = output_path.resolve()
    tmp = output_path.with_suffix(".tmp.json")
    write_holdout_manifest(
        holdout, tmp, meta={"split_seed": split_seed, "holdout_ratio": holdout_ratio}
    )
    _, holdout2 = apply_holdout_manifest(val_samples, tmp)
    keys1 = {sample_key(vd) for vd, _ in holdout}
    keys2 = {sample_key(vd) for vd, _ in holdout2}
    if keys1 != keys2:
        raise RuntimeError("manifest round-trip failed: holdout keys differ from stratified split")
    tmp.unlink(missing_ok=True)

    report = class_distribution_report(
        train_samples, val_samples, holdout, holdout_ratio=holdout_ratio
    )
    meta = {
        "split_seed": split_seed,
        "holdout_ratio": holdout_ratio,
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "created_utc": datetime.now(UTC).isoformat(),
        "distribution_report": report,
    }
    write_holdout_manifest(holdout, output_path, meta=meta)
    return report
