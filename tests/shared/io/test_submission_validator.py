"""Tests for ``validate_submission_csv``."""

from __future__ import annotations

from pathlib import Path

import pytest

from smth2smth.shared.io.submission import (
    SubmissionFormatError,
    validate_submission_csv,
    write_submission_csv,
)


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestValidSubmission:
    def test_well_formed_passes(self, tmp_path: Path) -> None:
        out = write_submission_csv(
            tmp_path / "sub.csv",
            video_names=["video_1", "video_2", "video_3"],
            predictions=[0, 5, 12],
        )
        report = validate_submission_csv(out, num_classes=33)
        assert report.num_rows == 3
        assert report.unique_videos == 3
        assert report.predictions == [0, 5, 12]
        assert report.video_names == ["video_1", "video_2", "video_3"]

    def test_expected_names_match(self, tmp_path: Path) -> None:
        out = write_submission_csv(
            tmp_path / "sub.csv",
            video_names=["video_a", "video_b"],
            predictions=[1, 2],
        )
        report = validate_submission_csv(out, expected_video_names={"video_a", "video_b"})
        assert report.num_rows == 2


class TestInvalidSubmission:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            validate_submission_csv(tmp_path / "missing.csv")

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.csv"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(SubmissionFormatError):
            validate_submission_csv(empty)

    def test_wrong_header_raises(self, tmp_path: Path) -> None:
        bad = _write(
            tmp_path / "bad.csv",
            ["video,prediction", "video_1,0"],
        )
        with pytest.raises(SubmissionFormatError):
            validate_submission_csv(bad)

    def test_non_integer_prediction_raises(self, tmp_path: Path) -> None:
        bad = _write(
            tmp_path / "bad.csv",
            ["video_name,predicted_class", "video_1,abc"],
        )
        with pytest.raises(SubmissionFormatError):
            validate_submission_csv(bad)

    def test_out_of_range_prediction_raises(self, tmp_path: Path) -> None:
        bad = _write(
            tmp_path / "bad.csv",
            ["video_name,predicted_class", "video_1,42"],
        )
        with pytest.raises(SubmissionFormatError):
            validate_submission_csv(bad, num_classes=10)

    def test_duplicate_video_name_raises(self, tmp_path: Path) -> None:
        bad = _write(
            tmp_path / "bad.csv",
            ["video_name,predicted_class", "video_1,0", "video_1,1"],
        )
        with pytest.raises(SubmissionFormatError):
            validate_submission_csv(bad)

    def test_unexpected_video_set_raises(self, tmp_path: Path) -> None:
        bad = _write(
            tmp_path / "bad.csv",
            ["video_name,predicted_class", "video_1,0", "video_99,2"],
        )
        with pytest.raises(SubmissionFormatError):
            validate_submission_csv(bad, expected_video_names={"video_1", "video_2"})

    def test_row_with_too_few_cells_raises(self, tmp_path: Path) -> None:
        bad = _write(
            tmp_path / "bad.csv",
            ["video_name,predicted_class", "video_1"],
        )
        with pytest.raises(SubmissionFormatError):
            validate_submission_csv(bad)
