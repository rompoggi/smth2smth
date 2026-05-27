#!/usr/bin/env python3
"""Compare top-1 of a train-only checkpoint on local train vs SSv2 extras (0.4→4 frames).

Extras are official SSv2 train+validation IDs not present in ``data/train`` (and optionally
``data/val``). Frames are synthesized from ``.webm`` with ``extract_professor_frames``
(``source_fraction=0.4``) + eval resize, matching the planned ``extended_train`` recipe.

Usage::

    PYTHONPATH=src uv run python scripts/eval_extras_vs_local.py \\
        --checkpoint checkpoints/track_a/round3_collected/arch2-perceiver-q16-trainonly.final-ep50.pt

    # Quick smoke (subsample)
    PYTHONPATH=src uv run python scripts/eval_extras_vs_local.py --max-extras 512 --max-train 2000
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from smth2smth.shared.data import (  # noqa: E402
    VideoFrameDataset,
    build_transforms,
    collect_video_samples,
    parse_class_index,
    pick_frame_indices,
)
from smth2smth.shared.data.ssv2_extended import (  # noqa: E402
    collect_target_video_ids,
    decode_video_frames,
    extract_professor_frames,
    local_class_dirs,
)
from smth2smth.shared.engine import EpochStats, evaluate_epoch  # noqa: E402
from smth2smth.shared.io.checkpoints import load_checkpoint  # noqa: E402
from smth2smth.shared.models import build_model  # noqa: E402


@dataclass(frozen=True)
class WebmSample:
    """One extra clip to evaluate from disk."""

    video_id: str
    label: int
    webm_path: Path


class WebmProfessorDataset(Dataset):
    """On-the-fly 4-frame tensors from SSv2 ``.webm`` (professor 0.4 window)."""

    def __init__(
        self,
        samples: list[WebmSample],
        *,
        num_frames: int,
        image_size: int,
        source_fraction: float,
        transform,
    ) -> None:
        self.samples = samples
        self.num_frames = num_frames
        self.image_size = image_size
        self.source_fraction = source_fraction
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.samples[index]
        decoded = decode_video_frames(item.webm_path)
        prof = extract_professor_frames(
            decoded,
            num_frames=4,
            source_fraction=self.source_fraction,
        )
        # Match VideoFrameDataset: 4 on-disk frames → pick_frame_indices(4, T)
        indices = pick_frame_indices(len(prof), self.num_frames)
        raw_frames = [prof[i] for i in indices]
        resized = [F.resize(img.convert("RGB"), [self.image_size, self.image_size]) for img in raw_frames]
        try:
            transformed = self.transform(resized)
            if isinstance(transformed, list):
                frames = transformed
            else:
                frames = [self.transform(frame) for frame in resized]  # type: ignore[arg-type]
        except Exception:
            frames = [self.transform(frame) for frame in resized]  # type: ignore[arg-type]
        video = torch.stack(frames, dim=0)
        return video, torch.tensor(item.label, dtype=torch.long)


def _class_index_map(train_dir: Path) -> dict[str, int]:
    """Map ``NNN_Class`` folder → class index (numeric prefix)."""
    out: dict[str, int] = {}
    for class_dir in local_class_dirs(train_dir):
        idx = parse_class_index(class_dir.name)
        if idx is not None:
            out[class_dir.name] = idx
    return out


def _collect_local_ids(train_dir: Path, val_dir: Path | None) -> set[str]:
    vid_re = re.compile(r"video_(\d+)$")
    ids: set[str] = set()
    for root in (train_dir, val_dir):
        if root is None or not root.is_dir():
            continue
        for class_dir in local_class_dirs(root):
            for vd in class_dir.iterdir():
                m = vid_re.match(vd.name)
                if m and vd.is_dir():
                    ids.add(m.group(1))
    return ids


def _build_extra_samples(
    *,
    videos_dir: Path,
    id_to_class: dict[str, str],
    folder_to_idx: dict[str, int],
    local_ids: set[str],
    max_extras: int | None,
    seed: int,
) -> list[WebmSample]:
    extras: list[WebmSample] = []
    for vid, folder in sorted(id_to_class.items()):
        if vid in local_ids:
            continue
        webm = videos_dir / f"{vid}.webm"
        if not webm.is_file():
            continue
        label = folder_to_idx.get(folder)
        if label is None:
            continue
        extras.append(WebmSample(video_id=vid, label=label, webm_path=webm))
    if max_extras is not None and len(extras) > max_extras:
        rng = random.Random(seed)
        extras = rng.sample(extras, max_extras)
    return extras


def _eval_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
) -> EpochStats:
    return evaluate_epoch(
        model,
        loader,
        nn.CrossEntropyLoss(),
        device,
        amp_enabled=amp,
        amp_dtype=torch.bfloat16,
    )


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
    parser.add_argument("--max-train", type=int, default=None, help="Subsample local train clips.")
    parser.add_argument("--max-extras", type=int, default=None, help="Subsample extra clips.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "outputs/ssv2_extended/extras_vs_local_eval.json")
    args = parser.parse_args()

    ck = load_checkpoint(args.checkpoint, map_location="cpu")
    cfg = OmegaConf.create(ck["config"])
    num_frames = int(cfg.dataset.num_frames)
    image_size = int(cfg.dataset.image_size)
    use_imagenet = bool(cfg.dataset.get("use_imagenet_norm", True))
    augment_cfg = cfg.get("augment")

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

    folder_to_idx = _class_index_map(args.train_dir)
    train_json = args.ssv2_annotations_dir / "something-something-v2-train.json"
    val_json = args.ssv2_annotations_dir / "something-something-v2-validation.json"
    ref_dirs = local_class_dirs(args.train_dir)
    id_to_class = collect_target_video_ids(train_json, val_json, ref_dirs)

    # Extras = SSv2 JSON ids not in professor train+val (same as overlap CSV "extra" pool).
    local_ids = _collect_local_ids(args.train_dir, args.val_dir)

    videos_dir = args.ssv2_videos_dir
    if not any(videos_dir.glob("*.webm")):
        nested = videos_dir / "20bn-something-something-v2"
        if nested.is_dir():
            videos_dir = nested

    train_samples = collect_video_samples(args.train_dir)
    if args.max_train is not None and len(train_samples) > args.max_train:
        rng = random.Random(args.seed)
        train_samples = rng.sample(train_samples, args.max_train)

    extra_samples = _build_extra_samples(
        videos_dir=videos_dir,
        id_to_class=id_to_class,
        folder_to_idx=folder_to_idx,
        local_ids=local_ids,
        max_extras=args.max_extras,
        seed=args.seed,
    )

    # Variant-template classes (professor kept base template only)
    variant_folders = {
        "005_Holding_something",
        "012_Pouring_something_into_something",
        "024_Putting_something_onto_something",
        "029_Throwing_something",
    }
    extra_variant = [s for s in extra_samples if id_to_class.get(s.video_id) in variant_folders]
    extra_other = [s for s in extra_samples if id_to_class.get(s.video_id) not in variant_folders]

    amp = device.type == "cuda"
    pin = device.type == "cuda"

    train_ds = VideoFrameDataset(
        args.train_dir,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=train_samples,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    def _extras_loader(samples: list[WebmSample]) -> DataLoader:
        ds = WebmProfessorDataset(
            samples,
            num_frames=num_frames,
            image_size=image_size,
            source_fraction=args.source_fraction,
            transform=eval_transform,
        )
        # Decode in workers; keep workers low to avoid RAM spikes on long clips
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(args.num_workers, 4),
            pin_memory=pin,
        )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"T={num_frames}  source_fraction={args.source_fraction}  device={device}")
    print(f"Local train clips: {len(train_samples)}  |  extras: {len(extra_samples)}")
    print(f"  extras (4 template-variant classes): {len(extra_variant)}")
    print(f"  extras (other classes): {len(extra_other)}")

    train_stats = _eval_loader(model, train_loader, device, amp=amp)
    print(f"\n[local train] top1={train_stats.top1:.4f}  top5={train_stats.top5:.4f}  loss={train_stats.loss:.4f}")

    results: dict = {
        "checkpoint": str(args.checkpoint),
        "source_fraction": args.source_fraction,
        "num_frames": num_frames,
        "n_train": len(train_samples),
        "n_extras": len(extra_samples),
        "local_train": {"top1": train_stats.top1, "top5": train_stats.top5, "loss": train_stats.loss},
    }

    if extra_samples:
        extra_stats = _eval_loader(model, _extras_loader(extra_samples), device, amp=amp)
        print(
            f"[extras 0.4→4] top1={extra_stats.top1:.4f}  top5={extra_stats.top5:.4f}  "
            f"loss={extra_stats.loss:.4f}"
        )
        results["extras_all"] = {
            "top1": extra_stats.top1,
            "top5": extra_stats.top5,
            "loss": extra_stats.loss,
        }
    if extra_variant:
        v_stats = _eval_loader(model, _extras_loader(extra_variant), device, amp=amp)
        print(f"[extras variant-template] top1={v_stats.top1:.4f}  n={len(extra_variant)}")
        results["extras_variant_templates"] = {"top1": v_stats.top1, "n": len(extra_variant)}
    if extra_other:
        o_stats = _eval_loader(model, _extras_loader(extra_other), device, amp=amp)
        print(f"[extras other classes] top1={o_stats.top1:.4f}  n={len(extra_other)}")
        results["extras_other"] = {"top1": o_stats.top1, "n": len(extra_other)}

    delta = results.get("extras_all", {}).get("top1", 0) - train_stats.top1
    print(f"\nΔ top1 (extras − train): {delta:+.4f}")
    results["delta_top1_extras_minus_train"] = delta

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
