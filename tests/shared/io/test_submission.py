"""Tests for ``smth2smth.shared.io.submission``."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from smth2smth.shared.io.submission import (
    DuplicateVideoFolderError,
    discover_all_test_videos,
    index_video_folders,
    load_manifest_video_names,
    resolve_video_dirs,
    write_submission_csv,
)


@pytest.fixture
def test_root(tmp_path: Path) -> Path:
    """Build a minimal test-folder layout::

    test/
      video_1/
        frame_000.jpg
      video_2/
        frame_000.jpg
      video_10/  # tests numeric-vs-string sort
        frame_000.jpg
    """
    root = tmp_path / "test"
    for name in ("video_1", "video_2", "video_10"):
        (root / name).mkdir(parents=True)
        (root / name / "frame_000.jpg").write_bytes(b"x")
    return root


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    p = tmp_path / "manifest.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_name"])
        for name in ("video_2", "video_1", "video_10"):
            writer.writerow([name])
    return p


class TestIndexAndDiscovery:
    def test_index_finds_all_video_folders(self, test_root: Path) -> None:
        index = index_video_folders(test_root)
        assert set(index.keys()) == {"video_1", "video_2", "video_10"}

    def test_discover_returns_sorted_by_string(self, test_root: Path) -> None:
        names, dirs = discover_all_test_videos(test_root)
        assert names == sorted(names)
        assert all(d.exists() for d in dirs)

    def test_duplicate_folder_names_raise(self, tmp_path: Path) -> None:
        root = tmp_path / "dup"
        (root / "a" / "video_1").mkdir(parents=True)
        (root / "b" / "video_1").mkdir(parents=True)
        with pytest.raises(DuplicateVideoFolderError):
            index_video_folders(root)


class TestManifestResolution:
    def test_load_manifest_preserves_order(self, manifest_path: Path) -> None:
        names = load_manifest_video_names(manifest_path)
        assert names == ["video_2", "video_1", "video_10"]

    def test_resolve_video_dirs_preserves_manifest_order(
        self, manifest_path: Path, test_root: Path
    ) -> None:
        names = load_manifest_video_names(manifest_path)
        dirs = resolve_video_dirs(test_root, names)
        assert [d.name for d in dirs] == names

    def test_missing_video_raises(self, test_root: Path) -> None:
        with pytest.raises(FileNotFoundError):
            resolve_video_dirs(test_root, ["video_404"])

    def test_manifest_without_column_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.csv"
        bad.write_text("foo\n1\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_manifest_video_names(bad)


class TestWriteSubmission:
    def test_writes_header_and_rows(self, tmp_path: Path) -> None:
        out = write_submission_csv(
            tmp_path / "sub.csv",
            video_names=["video_3", "video_1"],
            predictions=[7, 12],
        )
        with out.open(encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["video_name", "predicted_class"]
        assert rows[1] == ["video_3", "7"]
        assert rows[2] == ["video_1", "12"]

    def test_length_mismatch_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            write_submission_csv(
                tmp_path / "sub.csv",
                video_names=["a", "b"],
                predictions=[1],
            )

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested = tmp_path / "nested" / "deeper" / "sub.csv"
        write_submission_csv(nested, ["a"], [1])
        assert nested.is_file()
