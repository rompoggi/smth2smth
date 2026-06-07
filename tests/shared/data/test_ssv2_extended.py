"""Tests for ``smth2smth.shared.data.ssv2_extended``."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from smth2smth.shared.data.ssv2_extended import (
    TARGET_CLASS_DIR_NAMES,
    LocalFrameRef,
    build_class_count_table,
    collect_target_video_ids,
    compute_dhash,
    extract_professor_frames,
    extract_uniform_frames,
    filter_records_to_target_classes,
    hamming_distance,
    load_ssv2_records,
    map_template_to_local_folder,
    ssv2_clip_overlaps_local,
)


def test_target_classes_exclude_027() -> None:
    assert len(TARGET_CLASS_DIR_NAMES) == 32
    assert not any("027" in name for name in TARGET_CLASS_DIR_NAMES)


def test_collect_target_video_ids_unions_train_and_val(tmp_path: Path) -> None:
    (tmp_path / "000_Closing_something").mkdir()
    local_dirs = [tmp_path / "000_Closing_something"]
    annot = tmp_path / "annot"
    annot.mkdir()
    train_json = annot / "train.json"
    val_json = annot / "val.json"
    train_json.write_text(json.dumps([{"id": "1", "template": "Closing [something]"}]))
    val_json.write_text(json.dumps([{"id": "2", "template": "Closing [something]"}]))
    assert collect_target_video_ids(train_json, val_json, local_dirs) == {
        "1": "000_Closing_something",
        "2": "000_Closing_something",
    }


def test_filter_records_matches_template(tmp_path: Path) -> None:
    (tmp_path / "000_Closing_something").mkdir()
    local_dirs = [tmp_path / "000_Closing_something"]
    records = load_ssv2_records(
        _write_json(
            tmp_path,
            [
                {"id": "1", "template": "Closing [something]"},
                {"id": "2", "template": "Throwing [something]"},
            ],
        )
    )
    out = filter_records_to_target_classes(records, local_dirs)
    assert out == {"1": "000_Closing_something"}


def test_extract_professor_frames_first_60_percent() -> None:
    frames = [Image.new("RGB", (4, 4), color=(i, 0, 0)) for i in range(10)]
    picked = extract_professor_frames(frames, num_frames=4, source_fraction=0.6)
    assert len(picked) == 4
    # window = 6 frames (0..5); indices must not exceed 5
    assert all(p.getpixel((0, 0))[0] <= 5 for p in picked)


def test_extract_uniform_16_spans_full_clip() -> None:
    frames = [Image.new("RGB", (2, 2)) for _ in range(20)]
    uni = extract_uniform_frames(frames, 16)
    assert len(uni) == 16


def test_dhash_identical_images_zero_distance() -> None:
    img = Image.new("RGB", (32, 32), color=(128, 64, 32))
    a = compute_dhash(img)
    b = compute_dhash(img)
    assert hamming_distance(a, b) == 0


def test_ssv2_overlap_detects_duplicate_frame() -> None:
    img = Image.new("RGB", (16, 16), color=(10, 20, 30))
    dh = compute_dhash(img)
    local_refs = [LocalFrameRef(video_dir=Path("video_1"), frame_idx=0, dhash=dh)]
    decoded = [Image.new("RGB", (16, 16), color=(0, 0, 0))] * 5 + [img] + [img] * 5
    hit, reason = ssv2_clip_overlaps_local(decoded, local_refs, max_hamming=0)
    assert hit
    assert reason is not None


def test_class_count_table_subset_flag(tmp_path: Path) -> None:
    train = tmp_path / "train"
    val = tmp_path / "val"
    (train / "000_Closing_something" / "video_a").mkdir(parents=True)
    (val / "000_Closing_something" / "video_b").mkdir(parents=True)
    annot = tmp_path / "annot"
    annot.mkdir()
    train_json = annot / "something-something-v2-train.json"
    val_json = annot / "something-something-v2-validation.json"
    train_json.write_text(
        json.dumps([{"id": str(i), "template": "Closing [something]"} for i in range(10)])
    )
    val_json.write_text(
        json.dumps([{"id": str(i + 100), "template": "Closing [something]"} for i in range(5)])
    )
    rows = build_class_count_table(train, val, train_json, val_json)
    row = next(r for r in rows if r.local_folder == "000_Closing_something")
    assert row.local_total == 2
    assert row.ssv2_train_val_json == 15
    assert row.is_subset is True


def _write_json(tmp_path: Path, data: list) -> Path:
    p = tmp_path / "labels.json"
    p.write_text(json.dumps(data))
    return p


def test_map_template_to_local_folder_truncated_name(tmp_path: Path) -> None:
    name = "015_Pretending_to_pour_something_out_of_something_but_something_"
    (tmp_path / name).mkdir()
    folder = map_template_to_local_folder(
        "Pretending to pour [something] out of [something], but [something] is empty",
        [tmp_path / name],
    )
    assert folder == name
