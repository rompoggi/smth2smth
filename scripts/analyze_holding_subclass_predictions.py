#!/usr/bin/env python3
"""Per–SSv2-template prediction breakdown for the Holding family (official ids 16–20).

All map to professor folder ``005_Holding_something`` (train label 5), but SSv2
defines five distinct templates. Reports whether the 32-class model still
predicts class 5, and where errors go.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/analyze_holding_subclass_predictions.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from smth2smth.shared.data import VideoFrameDataset, build_transforms, parse_class_index  # noqa: E402
from smth2smth.shared.data.ssv2_extended import (  # noqa: E402
    collect_target_video_ids,
    load_ssv2_records,
    local_class_dirs,
)
from smth2smth.shared.engine import predict_argmax  # noqa: E402
from smth2smth.shared.io.checkpoints import load_checkpoint  # noqa: E402
from smth2smth.shared.models import build_model  # noqa: E402

HOLDING_FOLDER = "005_Holding_something"
HOLDING_IDX = 5

# Official SSv2 template strings (labels.json keys) → id
HOLDING_TEMPLATES: dict[str, int] = {
    "Holding something": 16,
    "Holding something behind something": 17,
    "Holding something in front of something": 18,
    "Holding something next to something": 19,
    "Holding something over something": 20,
}


def _load_eval_helpers():
    spec = importlib.util.spec_from_file_location(
        "eval_extras_vs_local", REPO_ROOT / "scripts" / "eval_extras_vs_local.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _bracket_template(plain: str) -> str:
    return plain.replace("something", "[something]")


def _short_idx_name(train_dir: Path, idx: int) -> str:
    for d in local_class_dirs(train_dir):
        if parse_class_index(d.name) == idx:
            return d.name.split("_", 1)[0] + " " + d.name.split("_", 1)[1][:28]
    return str(idx)


def _collect_local_holding(train_dir: Path, val_dir: Path) -> list[tuple[Path, str]]:
    """``(video_dir, video_id)`` under 005 folder."""
    out: list[tuple[Path, str]] = []
    vid_re = re.compile(r"video_(\d+)$")
    for root in (train_dir, val_dir):
        cd = root / HOLDING_FOLDER
        if not cd.is_dir():
            continue
        for vd in sorted(cd.iterdir()):
            m = vid_re.match(vd.name)
            if m and vd.is_dir():
                out.append((vd, m.group(1)))
    return out


def _template_for_id(video_id: str, template_by_id: dict[str, str]) -> str:
    """Plain-text template key matching labels.json."""
    raw = template_by_id.get(video_id, "")
    return raw.replace("[something]", "something").strip()


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
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs/ssv2_extended/holding_subclass_preds")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    eval_mod = _load_eval_helpers()
    WebmProfessorDataset = eval_mod.WebmProfessorDataset
    WebmSample = eval_mod.WebmSample
    _collect_local_ids = eval_mod._collect_local_ids

    ck = load_checkpoint(args.checkpoint, map_location="cpu")
    cfg = OmegaConf.create(ck["config"])
    num_frames = int(cfg.dataset.num_frames)
    image_size = int(cfg.dataset.image_size)
    use_imagenet = bool(cfg.dataset.get("use_imagenet_norm", True))
    augment_cfg = cfg.get("augment")
    num_classes = int(cfg.model.num_classes)

    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()

    eval_transform = build_transforms(
        image_size=image_size,
        is_training=False,
        use_imagenet_norm=use_imagenet,
        augment=OmegaConf.to_container(augment_cfg, resolve=True) if augment_cfg is not None else None,
    )

    template_by_id: dict[str, str] = {}
    for name in ("something-something-v2-train.json", "something-something-v2-validation.json"):
        for rec in load_ssv2_records(args.ssv2_annotations_dir / name):
            template_by_id[rec.video_id] = rec.template

    videos_dir = args.ssv2_videos_dir
    if not any(videos_dir.glob("*.webm")):
        nested = videos_dir / "20bn-something-something-v2"
        if nested.is_dir():
            videos_dir = nested

    local_ids = _collect_local_ids(args.train_dir, args.val_dir)
    local_holding = _collect_local_holding(args.train_dir, args.val_dir)

    # --- Local professor clips (JPEG on disk) ---
    local_samples = [(vd, HOLDING_IDX) for vd, _ in local_holding]
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
    local_preds, local_labels = predict_argmax(model, local_loader, device)
    local_preds = np.array(local_preds)
    local_ids_order = [vid for _, vid in local_holding]

    # --- All SSv2 holding-template clips on disk (extras + verify locals) ---
    train_json = args.ssv2_annotations_dir / "something-something-v2-train.json"
    val_json = args.ssv2_annotations_dir / "something-something-v2-validation.json"
    ref_dirs = local_class_dirs(args.train_dir)
    id_to_class = collect_target_video_ids(train_json, val_json, ref_dirs)

    webm_samples: list = []
    for vid, folder in id_to_class.items():
        if folder != HOLDING_FOLDER:
            continue
        plain = _template_for_id(vid, template_by_id)
        if plain not in HOLDING_TEMPLATES:
            continue
        webm = videos_dir / f"{vid}.webm"
        if webm.is_file():
            webm_samples.append(WebmSample(video_id=vid, label=HOLDING_IDX, webm_path=webm))

    webm_ds = WebmProfessorDataset(
        webm_samples,
        num_frames=num_frames,
        image_size=image_size,
        source_fraction=args.source_fraction,
        transform=eval_transform,
    )
    webm_loader = DataLoader(
        webm_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(args.num_workers, 4),
        pin_memory=device.type == "cuda",
    )
    webm_preds, webm_labels = predict_argmax(model, webm_loader, device)
    webm_preds = np.array(webm_preds)
    webm_ids = [s.video_id for s in webm_samples]

    rows: list[dict] = []

    def _record(split: str, vid: str, pred: int, source: str) -> None:
        plain = _template_for_id(vid, template_by_id)
        ssv2_id = HOLDING_TEMPLATES.get(plain)
        rows.append(
            {
                "video_id": vid,
                "split": split,
                "source": source,
                "ssv2_template": plain,
                "ssv2_label_id": ssv2_id,
                "gt_train_idx": HOLDING_IDX,
                "pred_train_idx": int(pred),
                "correct_coarse": int(pred == HOLDING_IDX),
                "in_professor_local": vid in local_ids,
            }
        )

    for vid, pred in zip(local_ids_order, local_preds):
        _record("local_jpeg", vid, int(pred), "professor_jpeg")

    for vid, pred in zip(webm_ids, webm_preds):
        if vid in local_ids:
            src = "ssv2_webm_local_overlap"
        else:
            src = "ssv2_webm_extra"
        _record("ssv2_webm_04", vid, int(pred), src)

    df = pd.DataFrame(rows)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "holding_per_video_predictions.csv", index=False)

    source_rank = {"professor_jpeg": 0, "ssv2_webm_local_overlap": 1, "ssv2_webm_extra": 2}
    deduped = (
        df.assign(_src_rank=df["source"].map(source_rank).fillna(9))
        .sort_values(["video_id", "_src_rank"])
        .drop_duplicates(subset=["video_id"], keep="first")
        .drop(columns=["_src_rank"])
    )

    # --- Summary per official template ---
    idx_names = {i: _short_idx_name(args.train_dir, i) for i in range(num_classes)}

    print(f"Checkpoint: {args.checkpoint.name}")
    print(f"Coarse GT for all: train class {HOLDING_IDX} ({HOLDING_FOLDER})\n")
    print("=" * 72)
    print("Per official SSv2 holding template (ids 16–20)")
    print("=" * 72)

    summary_rows = []
    for plain, ssv2_id in sorted(HOLDING_TEMPLATES.items(), key=lambda x: x[1]):
        sub = deduped[deduped["ssv2_template"] == plain]
        if sub.empty:
            continue
        n = len(sub)
        pred_counts = Counter(sub["pred_train_idx"])
        pct_5 = 100.0 * pred_counts.get(HOLDING_IDX, 0) / n
        top_wrong = [(p, c) for p, c in pred_counts.most_common(5) if p != HOLDING_IDX]

        local_n = int(sub["in_professor_local"].sum())
        extra_n = n - local_n

        print(f"\n[{ssv2_id}] {plain}")
        print(f"  n={n}  (professor local={local_n}, SSv2-only={extra_n})")
        print(f"  predicts coarse class 5: {pred_counts[HOLDING_IDX]}/{n} ({pct_5:.1f}%)")
        if top_wrong:
            print("  top other predictions:")
            for p, c in top_wrong[:4]:
                print(f"    -> class {p:2d} ({idx_names.get(p, '?')[:36]}): {c} ({100*c/n:.1f}%)")

        summary_rows.append(
            {
                "ssv2_id": ssv2_id,
                "template": plain,
                "n": n,
                "n_local": local_n,
                "n_extra": extra_n,
                "pct_pred_class_5": pct_5,
                "top_wrong_class": top_wrong[0][0] if top_wrong else None,
                "top_wrong_pct": 100.0 * top_wrong[0][1] / n if top_wrong else 0.0,
            }
        )

    # Can the model separate sub-types? (only within clips predicted as 5)
    print("\n" + "=" * 72)
    print("Do different holding templates get different WRONG-class patterns?")
    print("(Model has no fine-grained head — only 32 coarse classes.)")
    print("=" * 72)
    wrong = deduped.loc[deduped["correct_coarse"].eq(0)]
    if len(wrong):
        cross = pd.crosstab(wrong["ssv2_template"], wrong["pred_train_idx"])
        print(cross.to_string())

    print("\n" + "=" * 72)
    print("Local JPEG vs same-id SSv2 webm (0.4→4) — does pred change?")
    print("=" * 72)
    for vid in local_ids_order:
        j = df[(df["video_id"] == vid) & (df["source"] == "professor_jpeg")]
        w = df[(df["video_id"] == vid) & (df["source"].str.startswith("ssv2_webm"))]
        if j.empty or w.empty:
            continue
        pj, pw = int(j.iloc[0]["pred_train_idx"]), int(w.iloc[0]["pred_train_idx"])
        if pj != pw:
            print(f"  id {vid}: jpeg→{pj}  webm→{pw}  ({_template_for_id(vid, template_by_id)[:40]})")

    n_changed = 0
    n_compared = 0
    for vid in local_ids_order:
        j = df[(df["video_id"] == vid) & (df["source"] == "professor_jpeg")]
        w = df[(df["video_id"] == vid) & (df["source"].str.startswith("ssv2_webm"))]
        if j.empty or w.empty:
            continue
        n_compared += 1
        if int(j.iloc[0]["pred_train_idx"]) != int(w.iloc[0]["pred_train_idx"]):
            n_changed += 1
    if n_compared:
        print(f"\n  Prediction changed jpeg vs webm on {n_changed}/{n_compared} local holds "
              f"({100*n_changed/n_compared:.1f}%)")

    (out_dir / "holding_template_summary.json").write_text(
        json.dumps(summary_rows, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {out_dir}/holding_per_video_predictions.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
