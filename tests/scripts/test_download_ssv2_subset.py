"""Tests for ``scripts/download_ssv2_subset_4frame.py`` (E6).

Covers the pure, decode-free helpers: name-aligned class selection, the
first-fraction frame sampler, the on-disk frame writer, and the SSv2 label
loader. The webm-decode path needs PyAV + source data and is not exercised.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "download_ssv2_subset_4frame.py"
    spec = importlib.util.spec_from_file_location("download_ssv2_subset_4frame", script_path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["download_ssv2_subset_4frame"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_module()


def test_select_class_targets_matches_by_normalized_template(mod, tmp_path):
    for name in ("000_Closing_something", "001_Covering_something_with_something"):
        (tmp_path / name).mkdir()
    local_dirs = sorted(p for p in tmp_path.iterdir() if p.is_dir())
    records = [
        ("100", "Closing [something]"),  # -> 000_Closing_something
        ("101", "Covering [something] with [something]"),  # -> 001_...
        ("102", "Throwing [something]"),  # no local class -> dropped
    ]
    targets = mod.select_class_targets(local_dirs, records)
    assert targets == {
        "100": "000_Closing_something",
        "101": "001_Covering_something_with_something",
    }


def test_extract_real_frames_uses_leading_fraction(mod):
    frames = list(range(10))  # 0..9
    # source_fraction=0.6 -> window = first 6 frames (0..5); pick 4 via linspace.
    picked = mod.extract_real_frames(frames, num_frames=4, source_fraction=0.6)
    assert len(picked) == 4
    assert max(picked) <= 5, picked  # never sampled from the tail 40%
    assert picked == sorted(picked)  # temporal order preserved


def test_extract_real_frames_rejects_empty(mod):
    with pytest.raises(ValueError):
        mod.extract_real_frames([], num_frames=4)


def test_extract_real_frames_rejects_bad_fraction(mod):
    with pytest.raises(ValueError):
        mod.extract_real_frames([1, 2, 3], num_frames=2, source_fraction=1.5)


def test_write_clip_frames_layout(mod, tmp_path):
    frames = [Image.new("RGB", (8, 8), color=(i, i, i)) for i in range(4)]
    out = tmp_path / "000_Closing_something" / "video_100"
    n = mod.write_clip_frames(frames, out)
    assert n == 4
    written = sorted(p.name for p in out.glob("*.jpg"))
    assert written == ["frame_000.jpg", "frame_001.jpg", "frame_002.jpg", "frame_003.jpg"]


def test_load_ssv2_records_template_and_label_fallback(mod, tmp_path):
    labels = tmp_path / "train.json"
    labels.write_text(
        json.dumps(
            [
                {"id": "100", "template": "Closing [something]", "label": "Closing a door"},
                {"id": "101", "label": "Opening something"},  # no template -> label fallback
            ]
        )
    )
    records = mod.load_ssv2_records(labels)
    assert records == [("100", "Closing [something]"), ("101", "Opening something")]
