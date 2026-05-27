#!/usr/bin/env python3
"""Confusion-matrix comparison for SSv2 template-variant classes (005, 012, 024, 029).

Compares prediction patterns on professor local clips vs SSv2 extras (0.4→4 frames),
and breaks down extras by official SSv2 template string.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/analyze_variant_class_confusion.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from omegaconf import OmegaConf
from sklearn.metrics import classification_report, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# Reuse eval helpers from sibling script (import by path; not a package).
import importlib.util

_eval_spec = importlib.util.spec_from_file_location(
    "eval_extras_vs_local",
    REPO_ROOT / "scripts" / "eval_extras_vs_local.py",
)
_eval_mod = importlib.util.module_from_spec(_eval_spec)
assert _eval_spec.loader is not None
import sys as _sys

_sys.modules[_eval_spec.name] = _eval_mod
_eval_spec.loader.exec_module(_eval_mod)
WebmProfessorDataset = _eval_mod.WebmProfessorDataset
_build_extra_samples = _eval_mod._build_extra_samples
_class_index_map = _eval_mod._class_index_map
_collect_local_ids = _eval_mod._collect_local_ids
from smth2smth.shared.data import VideoFrameDataset, build_transforms, parse_class_index  # noqa: E402
from smth2smth.shared.data.ssv2_extended import (  # noqa: E402
    collect_target_video_ids,
    load_ssv2_records,
    local_class_dirs,
)
from smth2smth.shared.engine import predict_argmax  # noqa: E402
from smth2smth.shared.io.checkpoints import load_checkpoint  # noqa: E402
from smth2smth.shared.models import build_model  # noqa: E402

VARIANT_FOLDERS: tuple[str, ...] = (
    "005_Holding_something",
    "012_Pouring_something_into_something",
    "024_Putting_something_onto_something",
    "029_Throwing_something",
)


def _short_template(t: str, max_len: int = 48) -> str:
    s = t.replace("[something]", "sth")
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _load_template_map(annot_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("something-something-v2-train.json", "something-something-v2-validation.json"):
        for rec in load_ssv2_records(annot_dir / name):
            out[rec.video_id] = rec.template
    return out


def _filter_samples(samples: list, folder_to_idx: dict[str, int], target_idx: set[int]):
    """Keep samples whose label is in ``target_idx``."""
    return [s for s in samples if (s[1] if isinstance(s, tuple) else s.label) in target_idx]


def _collect_local_variant_samples(train_dir: Path, val_dir: Path, target_idx: set[int]):
    out = []
    for root in (train_dir, val_dir):
        if not root.is_dir():
            continue
        for class_dir in local_class_dirs(root):
            if class_dir.name not in VARIANT_FOLDERS:
                continue
            idx = parse_class_index(class_dir.name)
            if idx is None or idx not in target_idx:
                continue
            for vd in sorted(class_dir.iterdir()):
                if vd.is_dir() and vd.name.startswith("video_"):
                    out.append((vd, idx))
    return out


def _run_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    preds, labels = predict_argmax(model, loader, device)
    return np.array(preds), np.array(labels)


def _plot_cm(
    cm: np.ndarray,
    labels: list[str],
    *,
    title: str,
    path: Path,
    normalize: bool,
) -> None:
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        display = np.divide(cm, row_sums, where=row_sums > 0)
        fmt = ".2f"
        cbar_label = "Row recall"
    else:
        display = cm
        fmt = "d"
        cbar_label = "Count"
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        vmin=0,
        vmax=1 if normalize else None,
        cbar_kws={"label": cbar_label},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen–Shannon divergence between two discrete distributions."""
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT
        / "checkpoints/track_a/round3_collected/arch2-perceiver-q16-trainonly.final-ep50.pt",
    )
    parser.add_argument("--train-dir", type=Path, default=REPO_ROOT / "data/train")
    parser.add_argument("--val-dir", type=Path, default=REPO_ROOT / "data/val")
    parser.add_argument(
        "--ssv2-videos-dir",
        type=Path,
        default=REPO_ROOT / "data/ssv2/raw/20bn-something-something-v2",
    )
    parser.add_argument("--ssv2-annotations-dir", type=Path, default=REPO_ROOT / "data/ssv2/raw/annotations")
    parser.add_argument("--source-fraction", type=float, default=0.4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs/ssv2_extended/variant_class_cm")
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()

    target_idx = {parse_class_index(f) for f in VARIANT_FOLDERS}
    target_idx.discard(None)
    target_idx = {int(x) for x in target_idx}
    short_labels = ["005 Hold", "012 Pour→", "024 Put on", "029 Throw"]

    ck = load_checkpoint(args.checkpoint, map_location="cpu")
    cfg = OmegaConf.create(ck["config"])
    num_frames = int(cfg.dataset.num_frames)
    image_size = int(cfg.dataset.image_size)
    use_imagenet = bool(cfg.dataset.get("use_imagenet_norm", True))
    augment_cfg = cfg.get("augment")

    import torch
    from torch.utils.data import DataLoader

    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model_state_dict"], strict=True)

    eval_transform = build_transforms(
        image_size=image_size,
        is_training=False,
        use_imagenet_norm=use_imagenet,
        augment=OmegaConf.to_container(augment_cfg, resolve=True) if augment_cfg is not None else None,
    )

    folder_to_idx = _class_index_map(args.train_dir)
    train_json = args.ssv2_annotations_dir / "something-something-v2-train.json"
    val_json = args.ssv2_annotations_dir / "something-something-v2-validation.json"
    ref_dirs = local_class_dirs(args.train_dir)
    id_to_class = collect_target_video_ids(train_json, val_json, ref_dirs)
    template_by_id = _load_template_map(args.ssv2_annotations_dir)
    local_ids = _collect_local_ids(args.train_dir, args.val_dir)

    videos_dir = args.ssv2_videos_dir
    if not any(videos_dir.glob("*.webm")):
        nested = videos_dir / "20bn-something-something-v2"
        if nested.is_dir():
            videos_dir = nested

    # Local professor clips (train+val) in the 4 classes
    local_samples = _collect_local_variant_samples(args.train_dir, args.val_dir, target_idx)
    local_ds = VideoFrameDataset(
        args.train_dir,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=local_samples,
    )
    local_loader = DataLoader(
        local_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    extra_webm = _build_extra_samples(
        videos_dir=videos_dir,
        id_to_class=id_to_class,
        folder_to_idx=folder_to_idx,
        local_ids=local_ids,
        max_extras=None,
        seed=42,
    )
    extra_ds = WebmProfessorDataset(
        extra_webm,
        num_frames=num_frames,
        image_size=image_size,
        source_fraction=args.source_fraction,
        transform=eval_transform,
    )
    extra_loader = DataLoader(
        extra_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(args.num_workers, 4),
        pin_memory=device.type == "cuda",
    )

    print(f"Local variant clips: {len(local_samples)}  |  extras: {len(extra_webm)}")
    local_preds, local_labels = _run_predictions(model, local_loader, device)
    extra_preds, extra_labels = _run_predictions(model, extra_loader, device)

    # Map video_id → template for extras
    extra_ids = [s.video_id for s in extra_webm]
    extra_templates = [template_by_id.get(vid, "?") for vid in extra_ids]

    idx_order = sorted(target_idx)
    label_names = short_labels

    cm_local = confusion_matrix(local_labels, local_preds, labels=idx_order)
    cm_extra = confusion_matrix(extra_labels, extra_preds, labels=idx_order)

    # Row-normalized recall vectors (flattened) for similarity
    def _row_recall_flat(cm: np.ndarray) -> np.ndarray:
        rs = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, rs, where=rs > 0)
        return norm.flatten()

    js = _js_divergence(_row_recall_flat(cm_local), _row_recall_flat(cm_extra))
    local_top1 = float((local_preds == local_labels).mean())
    extra_top1 = float((extra_preds == extra_labels).mean())

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_cm(
        cm_local,
        label_names,
        title=f"Local professor (n={len(local_labels)}) — top1={local_top1:.1%}",
        path=out_dir / "cm_local_4class_counts.png",
        normalize=False,
    )
    _plot_cm(
        cm_local,
        label_names,
        title="Local — row-normalized recall",
        path=out_dir / "cm_local_4class_norm.png",
        normalize=True,
    )
    _plot_cm(
        cm_extra,
        label_names,
        title=f"SSv2 extras 0.4→4 (n={len(extra_labels)}) — top1={extra_top1:.1%}",
        path=out_dir / "cm_extras_4class_counts.png",
        normalize=False,
    )
    _plot_cm(
        cm_extra,
        label_names,
        title="Extras — row-normalized recall",
        path=out_dir / "cm_extras_4class_norm.png",
        normalize=True,
    )

    # Per-template breakdown on extras
    template_stats: dict[str, dict] = {}
    by_folder_template: dict[str, Counter] = defaultdict(Counter)
    for vid, true_l, pred, tmpl in zip(extra_ids, extra_labels, extra_preds, extra_templates):
        folder = id_to_class.get(vid, "?")
        key = _short_template(tmpl, 56)
        by_folder_template[folder][key] += 1
        if key not in template_stats:
            template_stats[key] = {"n": 0, "correct": 0, "true_class": int(true_l)}
        template_stats[key]["n"] += 1
        template_stats[key]["correct"] += int(pred == true_l)

    # Professor base templates (what local has)
    base_templates = {
        "005_Holding_something": "Holding [something]",
        "012_Pouring_something_into_something": "Pouring [something] into [something]",
        "024_Putting_something_onto_something": "Putting [something] onto [something]",
        "029_Throwing_something": "Throwing [something]",
    }

    rows = []
    for folder in VARIANT_FOLDERS:
        idx = parse_class_index(folder)
        loc_mask = local_labels == idx
        ext_mask = extra_labels == idx
        loc_acc = float((local_preds[loc_mask] == local_labels[loc_mask]).mean()) if loc_mask.any() else 0.0
        ext_acc = float((extra_preds[ext_mask] == extra_labels[ext_mask]).mean()) if ext_mask.any() else 0.0
        rows.append(
            {
                "folder": folder,
                "class_idx": idx,
                "n_local": int(loc_mask.sum()),
                "n_extras": int(ext_mask.sum()),
                "top1_local": loc_acc,
                "top1_extras": ext_acc,
                "professor_template": base_templates[folder],
            }
        )

    template_rows = []
    for folder in VARIANT_FOLDERS:
        for tmpl, cnt in by_folder_template[folder].most_common():
            st = template_stats.get(tmpl, {"n": cnt, "correct": 0})
            acc = st["correct"] / st["n"] if st["n"] else 0.0
            is_base = tmpl == _short_template(base_templates[folder], 56)
            template_rows.append(
                {
                    "folder": folder,
                    "template": tmpl,
                    "n": cnt,
                    "top1": acc,
                    "is_professor_base_template": is_base,
                }
            )

    report = {
        "checkpoint": str(args.checkpoint),
        "classes": list(VARIANT_FOLDERS),
        "class_indices": idx_order,
        "local_top1_4class": local_top1,
        "extras_top1_4class": extra_top1,
        "js_divergence_row_recall_4x4": js,
        "per_class": rows,
        "extras_by_template": template_rows,
        "sklearn_local": classification_report(
            local_labels, local_preds, labels=idx_order, target_names=label_names, output_dict=True
        ),
        "sklearn_extras": classification_report(
            extra_labels, extra_preds, labels=idx_order, target_names=label_names, output_dict=True
        ),
    }
    (out_dir / "variant_class_confusion_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"\n4-class top1  local={local_top1:.4f}  extras={extra_top1:.4f}")
    print(f"JS divergence (row-recall 4×4): {js:.4f}  (0=identical, >0.1=quite different)")
    print("\nPer-class:")
    for r in rows:
        print(
            f"  {r['class_idx']:02d} local {r['top1_local']:.1%} (n={r['n_local']})  "
            f"extras {r['top1_extras']:.1%} (n={r['n_extras']})"
        )
    print("\nExtras by SSv2 template (top1):")
    for tr in template_rows:
        tag = " [prof base]" if tr["is_professor_base_template"] else " [variant]"
        print(f"  {tr['folder'][:12]} | {tr['template']}{tag}: n={tr['n']} acc={tr['top1']:.1%}")

    print(f"\nWrote figures and {out_dir / 'variant_class_confusion_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
