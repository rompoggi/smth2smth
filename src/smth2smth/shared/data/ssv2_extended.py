"""SSv2 full-dataset helpers for building ``data/extended_train/``.

Pipeline stages (see ``scripts/build_extended_train.py``):

1. Filter official SSv2 train+validation JSON to our 32 local class folders.
2. Compare per-class counts against ``data/train`` and ``data/val``.
3. Drop SSv2 clips whose frames overlap local train/val (dHash on professor 4-frames
   and on GluonCV-style uniform 16-frame samples from the full decode).
4. Re-sample surviving clips with the professor recipe (first ~40% → 4 frames via
   :func:`pick_frame_indices`) and write ``NNN_Class/video_<id>/frame_*.jpg``.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from PIL import Image

from smth2smth.shared.data import parse_class_index
from smth2smth.shared.data.video_dataset import pick_frame_indices, _list_frame_paths
from smth2smth.track_b.zero_shot import normalize_class_name

# Our on-disk 32-class subset (folder names under data/train). Class index 27 is
# absent from both train and full SSv2 release folders.
TARGET_CLASS_DIR_NAMES: frozenset[str] = frozenset(
    {
        "000_Closing_something",
        "001_Covering_something_with_something",
        "002_Dropping_something_into_something",
        "003_Folding_something",
        "004_Hitting_something_with_something",
        "005_Holding_something",
        "006_Moving_something_away_from_something",
        "007_Moving_something_closer_to_something",
        "008_Moving_something_down",
        "009_Moving_something_up",
        "010_Opening_something",
        "011_Picking_something_up",
        "012_Pouring_something_into_something",
        "013_Pouring_something_out_of_something",
        "014_Pretending_to_pick_something_up",
        "015_Pretending_to_pour_something_out_of_something_but_something_",
        "016_Pretending_to_put_something_into_something",
        "017_Pretending_to_throw_something",
        "018_Pulling_something_from_left_to_right",
        "019_Pulling_something_from_right_to_left",
        "020_Putting_something_behind_something",
        "021_Putting_something_in_front_of_something",
        "022_Putting_something_into_something",
        "023_Putting_something_next_to_something",
        "024_Putting_something_onto_something",
        "025_Showing_something_to_the_camera",
        "026_Spilling_something_next_to_something",
        "028_Taking_something_out_of_something",
        "029_Throwing_something",
        "030_Turning_something_upside_down",
        "031_Uncovering_something",
        "032_Unfolding_something",
    }
)

SSV2_HF_REPO = "HuggingFaceM4/something_something_v2"
SSV2_ANNOTATION_FILES = (
    "something-something-v2-labels.json",
    "something-something-v2-train.json",
    "something-something-v2-validation.json",
    "something-something-v2-test.json",
)


@dataclass(frozen=True)
class Ssv2Record:
    """One SSv2 annotation row."""

    video_id: str
    template: str
    label_id: int | None = None


@dataclass
class ClassCountRow:
    """Per-class clip counts for subset comparison."""

    local_folder: str
    local_idx: int | None
    local_train: int
    local_val: int
    local_total: int
    ssv2_train_json: int
    ssv2_val_json: int
    ssv2_train_val_json: int
    is_subset: bool | None = None


@dataclass
class DedupeReport:
    """Summary of overlap filtering against local data."""

    candidates: int = 0
    removed_overlap: int = 0
    removed_reserved_ids: int = 0
    kept: int = 0
    examples: list[str] = field(default_factory=list)


def load_ssv2_records(labels_json: Path) -> list[Ssv2Record]:
    """Load SSv2 ``train.json`` / ``validation.json`` rows."""
    data = json.loads(labels_json.read_text(encoding="utf-8"))
    out: list[Ssv2Record] = []
    for entry in data:
        vid = str(entry["id"])
        template = str(entry.get("template") or "")
        if not template:
            raw_label = entry.get("label", "")
            template = str(raw_label) if raw_label is not None else ""
        label_id = entry.get("label")
        if isinstance(label_id, int):
            lid: int | None = label_id
        elif isinstance(label_id, str) and label_id.isdigit():
            lid = int(label_id)
        else:
            lid = None
        out.append(Ssv2Record(video_id=vid, template=template, label_id=lid))
    return out


def load_ssv2_labels_map(labels_json: Path) -> dict[int, str]:
    """Load ``labels.json`` id → template string."""
    data = json.loads(labels_json.read_text(encoding="utf-8"))
    return {int(row["id"]): str(row["name"]) for row in data}


def local_class_dirs(root: Path) -> list[Path]:
    """Sorted ``NNN_Class`` folders under ``root`` that are in our target set."""
    if not root.is_dir():
        return []
    return sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and p.name in TARGET_CLASS_DIR_NAMES
    )


def map_template_to_local_folder(template: str, local_dirs: list[Path]) -> str | None:
    """Map an SSv2 template string to a local folder name, if in our subset."""
    norm = normalize_class_name(template)
    local_by_norm = {normalize_class_name(d.name): d.name for d in local_dirs}
    if norm in local_by_norm:
        return local_by_norm[norm]
    # Filesystem-truncated folder names (e.g. 015_…_but_something_).
    matches = [
        name
        for ln, name in local_by_norm.items()
        if norm == ln or norm.startswith(ln + " ")
    ]
    return matches[0] if len(matches) == 1 else None


def filter_records_to_target_classes(
    records: Iterable[Ssv2Record],
    local_dirs: list[Path],
) -> dict[str, str]:
    """Return ``{video_id: local_folder}`` for clips in our 32 classes."""
    targets: dict[str, str] = {}
    for rec in records:
        folder = map_template_to_local_folder(rec.template, local_dirs)
        if folder is not None:
            targets[rec.video_id] = folder
    return targets


def collect_target_video_ids(
    ssv2_train_json: Path,
    ssv2_val_json: Path,
    local_dirs: list[Path],
) -> dict[str, str]:
    """Union of train+validation rows mapped to our 32 local class folders.

    Args:
        ssv2_train_json: Official ``train.json`` (or symlink).
        ssv2_val_json: Official ``validation.json``.
        local_dirs: Reference ``NNN_Class`` folders from ``local_class_dirs``.

    Returns:
        ``{video_id: local_folder}`` for every clip to keep when pruning raw
        ``.webm`` dumps to the challenge subset.
    """
    targets: dict[str, str] = {}
    if ssv2_train_json.is_file():
        targets.update(
            filter_records_to_target_classes(load_ssv2_records(ssv2_train_json), local_dirs)
        )
    if ssv2_val_json.is_file():
        for vid, folder in filter_records_to_target_classes(
            load_ssv2_records(ssv2_val_json), local_dirs
        ).items():
            targets.setdefault(vid, folder)
    return targets


def _count_videos_in_class_dir(class_dir: Path) -> int:
    return sum(1 for p in class_dir.iterdir() if p.is_dir() and p.name.startswith("video_"))


def count_local_clips_per_class(train_dir: Path, val_dir: Path) -> dict[str, tuple[int, int]]:
    """Count video folders per class in train and val."""
    counts: dict[str, tuple[int, int]] = {name: (0, 0) for name in sorted(TARGET_CLASS_DIR_NAMES)}
    if train_dir.is_dir():
        for class_dir in local_class_dirs(train_dir):
            tr, va = counts[class_dir.name]
            counts[class_dir.name] = (_count_videos_in_class_dir(class_dir), va)
    if val_dir.is_dir():
        for class_dir in local_class_dirs(val_dir):
            tr, va = counts[class_dir.name]
            counts[class_dir.name] = (tr, _count_videos_in_class_dir(class_dir))
    return counts


def count_ssv2_records_per_class(
    train_records: list[Ssv2Record],
    val_records: list[Ssv2Record],
    local_dirs: list[Path],
) -> dict[str, tuple[int, int]]:
    """Count SSv2 train/val JSON rows mapped to each local folder."""
    counts: dict[str, tuple[int, int]] = {name: (0, 0) for name in sorted(TARGET_CLASS_DIR_NAMES)}

    def _bump(records: list[Ssv2Record], slot: int) -> None:
        for rec in records:
            folder = map_template_to_local_folder(rec.template, local_dirs)
            if folder is None:
                continue
            tr, va = counts[folder]
            counts[folder] = (tr + 1, va) if slot == 0 else (tr, va + 1)

    _bump(train_records, 0)
    _bump(val_records, 1)
    return counts


def build_class_count_table(
    train_dir: Path,
    val_dir: Path,
    ssv2_train_json: Path,
    ssv2_val_json: Path,
) -> list[ClassCountRow]:
    """Compare local vs official SSv2 per-class instance counts."""
    ref_dirs = local_class_dirs(train_dir) or local_class_dirs(val_dir)
    local_counts = count_local_clips_per_class(train_dir, val_dir)
    ssv2_counts = count_ssv2_records_per_class(
        load_ssv2_records(ssv2_train_json),
        load_ssv2_records(ssv2_val_json),
        ref_dirs,
    )
    rows: list[ClassCountRow] = []
    for folder in sorted(TARGET_CLASS_DIR_NAMES):
        tr, va = local_counts.get(folder, (0, 0))
        st, sv = ssv2_counts.get(folder, (0, 0))
        local_idx = parse_class_index(folder)
        rows.append(
            ClassCountRow(
                local_folder=folder,
                local_idx=local_idx,
                local_train=tr,
                local_val=va,
                local_total=tr + va,
                ssv2_train_json=st,
                ssv2_val_json=sv,
                ssv2_train_val_json=st + sv,
                is_subset=(tr + va) <= (st + sv) if (st + sv) > 0 else None,
            )
        )
    return rows


def compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Difference hash (64-bit when ``hash_size=8``)."""
    gray = image.convert("L").resize(
        (hash_size + 1, hash_size),
        Image.Resampling.BILINEAR,
    )
    pixels = list(gray.getdata())
    bits = 0
    bit = 0
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            left = pixels[row_start + col]
            right = pixels[row_start + col + 1]
            if left > right:
                bits |= 1 << bit
            bit += 1
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer bit patterns."""
    return (a ^ b).bit_count()


def extract_professor_frames(
    decoded_frames: list[Image.Image],
    num_frames: int = 4,
    source_fraction: float = 0.4,
) -> list[Image.Image]:
    """First ``source_fraction`` of decode, then ``num_frames`` linspace picks."""
    n = len(decoded_frames)
    if n == 0:
        raise ValueError("decoded_frames is empty.")
    if not 0.0 < source_fraction <= 1.0:
        raise ValueError(f"source_fraction must be in (0, 1], got {source_fraction}.")
    window = max(1, int(round(n * source_fraction)))
    indices = pick_frame_indices(window, num_frames)
    return [decoded_frames[i] for i in indices]


def extract_uniform_frames(
    decoded_frames: list[Image.Image],
    num_frames: int = 16,
) -> list[Image.Image]:
    """Uniform linspace over the full clip (GluonCV / VideoMAE SSv2 style)."""
    n = len(decoded_frames)
    if n == 0:
        raise ValueError("decoded_frames is empty.")
    indices = pick_frame_indices(n, num_frames)
    return [decoded_frames[i] for i in indices]


def frame_dhashes(frames: Iterable[Image.Image]) -> list[int]:
    """dHash each frame."""
    return [compute_dhash(f) for f in frames]


@dataclass(frozen=True)
class LocalFrameRef:
    """Reference to one on-disk frame in our dataset."""

    video_dir: Path
    frame_idx: int
    dhash: int


def collect_local_frame_index(
    train_dir: Path,
    val_dir: Path,
    *,
    max_clips: int | None = None,
) -> list[LocalFrameRef]:
    """Index every JPEG frame in local train+val."""
    refs: list[LocalFrameRef] = []
    n_clips = 0
    for split in (train_dir, val_dir):
        if not split.is_dir():
            continue
        for class_dir in local_class_dirs(split):
            for video_dir in sorted(class_dir.iterdir()):
                if not video_dir.is_dir() or not video_dir.name.startswith("video_"):
                    continue
                if max_clips is not None and n_clips >= max_clips:
                    return refs
                n_clips += 1
                paths = _list_frame_paths(video_dir)
                for fi, fp in enumerate(paths):
                    with Image.open(fp) as im:
                        refs.append(
                            LocalFrameRef(
                                video_dir=video_dir,
                                frame_idx=fi,
                                dhash=compute_dhash(im.convert("RGB")),
                            )
                        )
    return refs


def ssv2_clip_overlaps_local(
    decoded_frames: list[Image.Image],
    local_refs: list[LocalFrameRef],
    *,
    max_hamming: int = 5,
    check_uniform_16: bool = True,
    professor_num_frames: int = 4,
    professor_source_fraction: float = 0.4,
) -> tuple[bool, str | None]:
    """True if any local frame matches this SSv2 source clip.

    Compares local on-disk frames against:

    * professor 4-frame extract (first 60% window), and
    * uniform 16-frame sample over the full decode (SSv2/GluonCV style).
    """
    if not local_refs:
        return False, None

    prof = extract_professor_frames(
        decoded_frames,
        num_frames=professor_num_frames,
        source_fraction=professor_source_fraction,
    )
    candidates = frame_dhashes(prof)
    if check_uniform_16:
        candidates.extend(frame_dhashes(extract_uniform_frames(decoded_frames, 16)))

    for dh in candidates:
        for ref in local_refs:
            if hamming_distance(dh, ref.dhash) <= max_hamming:
                return True, f"{ref.video_dir.name} frame {ref.frame_idx}"
    return False, None


def write_clip_frames(frames: list[Image.Image], out_video_dir: Path) -> int:
    """Write ``frame_NNN.jpg`` under ``out_video_dir``."""
    out_video_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.convert("RGB").save(out_video_dir / f"frame_{i:03d}.jpg", quality=95)
    return len(frames)


def decode_video_frames(video_path: Path) -> list[Image.Image]:
    """Decode all frames from a ``.webm`` clip via PyAV."""
    if not video_path.is_file():
        raise FileNotFoundError(f"SSv2 source video not found: {video_path}")
    try:
        import av  # type: ignore
    except ImportError as exc:
        raise ImportError("Decoding SSv2 clips requires PyAV (`uv add av`).") from exc

    frames: list[Image.Image] = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            frames.append(Image.fromarray(frame.to_ndarray(format="rgb24")))
    return frames


def reserved_ssv2_video_ids(
    ssv2_val_json: Path,
    ssv2_test_json: Path,
    *,
    exclude_all_official_val: bool = True,
) -> set[str]:
    """Video IDs we must not use for extended_train (val/test leakage guard)."""
    reserved: set[str] = set()
    if ssv2_test_json.is_file():
        reserved.update(r.video_id for r in load_ssv2_records(ssv2_test_json))
    if exclude_all_official_val and ssv2_val_json.is_file():
        reserved.update(r.video_id for r in load_ssv2_records(ssv2_val_json))
    return reserved


def download_ssv2_annotations(out_dir: Path) -> list[Path]:
    """Download official JSON annotations from Qualcomm (no videos)."""
    import shutil
    import zipfile

    out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = out_dir / "labels"
    if labels_dir.is_dir() and (labels_dir / "train.json").is_file():
        return [
            out_dir / "something-something-v2-train.json",
            out_dir / "something-something-v2-validation.json",
            out_dir / "something-something-v2-test.json",
            out_dir / "something-something-v2-labels.json",
        ]

    archive = out_dir.parent / "archives" / "20bn-something-something-download-package-labels.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        import urllib.request

        url = (
            "https://softwarecenter.qualcomm.com/api/download/software/dataset/"
            "AIDataset/Something-Something-V2/20bn-something-something-download-package-labels.zip"
        )
        urllib.request.urlretrieve(url, archive)

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(out_dir)

    mapping = {
        "train.json": "something-something-v2-train.json",
        "validation.json": "something-something-v2-validation.json",
        "test.json": "something-something-v2-test.json",
        "labels.json": "something-something-v2-labels.json",
    }
    written: list[Path] = []
    for src_name, dst_name in mapping.items():
        src = labels_dir / src_name
        dst = out_dir / dst_name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        shutil.copy2(src, dst)
        written.append(dst)
    return written
