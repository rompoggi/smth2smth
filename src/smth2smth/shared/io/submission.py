"""Submission file writer + test-folder discovery.

Replaces the inline helpers from the professor baseline's ``create_submission.py``
(``_index_video_folders``, ``resolve_video_dirs``, ``discover_all_test_videos``).

Output format (single CSV with header)::

    video_name,predicted_class
    video_12345,7
    video_45678,12
"""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

VIDEO_FOLDER_PREFIX: str = "video_"


class DuplicateVideoFolderError(ValueError):
    """Raised when two video folders share the same name."""


def index_video_folders(test_root: Path) -> dict[str, Path]:
    """Walk ``test_root`` once and map each ``video_<id>`` folder name to its path.

    Frames live inside ``video_<id>`` directories, so we prune the walk there
    instead of scanning every JPEG file.

    Args:
        test_root: Root directory of the test split.

    Returns:
        Dict mapping ``"video_<id>"`` -> resolved :class:`Path`.

    Raises:
        DuplicateVideoFolderError: If two video folders share the same name.
    """
    test_root = test_root.resolve()
    index: dict[str, Path] = {}
    for dirpath, dirs, _files in os.walk(test_root, topdown=True):
        base = Path(dirpath)
        for name in list(dirs):
            if not name.startswith(VIDEO_FOLDER_PREFIX):
                continue
            resolved = (base / name).resolve()
            if name in index:
                raise DuplicateVideoFolderError(
                    f"Duplicate video folder name {name!r}: {index[name]} and {resolved}"
                )
            index[name] = resolved
            # Prevent descending further; frames live directly under video_<id>.
            dirs.remove(name)
    return index


def discover_all_test_videos(test_root: Path) -> tuple[list[str], list[Path]]:
    """Discover all ``video_<id>`` folders under ``test_root`` sorted by name."""
    index = index_video_folders(test_root)
    names = sorted(index.keys())
    return names, [index[name] for name in names]


def load_manifest_video_names(manifest_path: Path) -> list[str]:
    """Read a manifest CSV that contains a ``video_name`` column.

    Args:
        manifest_path: CSV with a header row including ``video_name``.

    Returns:
        Ordered list of video names as they appear in the manifest.

    Raises:
        ValueError: If the file is missing or has no ``video_name`` column.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "video_name" not in reader.fieldnames:
            raise ValueError(f"{manifest_path} must contain a 'video_name' column.")
        return [row["video_name"].strip() for row in reader if row.get("video_name")]


def resolve_video_dirs(test_root: Path, video_names: Iterable[str]) -> list[Path]:
    """Map each manifest video name to its folder under ``test_root``.

    Raises:
        FileNotFoundError: If any manifest entry has no matching folder.
    """
    index = index_video_folders(test_root)
    out: list[Path] = []
    missing: list[str] = []
    for name in video_names:
        path = index.get(name)
        if path is None:
            missing.append(name)
        else:
            out.append(path)
    if missing:
        sample = ", ".join(repr(m) for m in missing[:5])
        extra = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"{len(missing)} manifest video(s) not found under {test_root}: {sample}{extra}"
        )
    return out


@dataclass
class SubmissionReport:
    """Result of validating a submission CSV.

    Attributes:
        path: Resolved path of the validated CSV.
        num_rows: Number of data rows (excluding header).
        unique_videos: Number of distinct ``video_name`` values.
        predictions: Parsed list of integer predictions in file order.
        video_names: Parsed list of video names in file order.
    """

    path: Path
    num_rows: int
    unique_videos: int
    predictions: list[int]
    video_names: list[str]


class SubmissionFormatError(ValueError):
    """Raised when a submission CSV violates the expected format."""


def validate_submission_csv(
    csv_path: Path,
    *,
    num_classes: int | None = None,
    expected_video_names: Iterable[str] | None = None,
) -> SubmissionReport:
    """Validate a submission CSV's format and content.

    Checks performed:
        * file exists and is non-empty,
        * header is exactly ``video_name,predicted_class``,
        * every data row has exactly two cells,
        * ``video_name`` values are unique,
        * ``predicted_class`` values are integers (and in ``[0, num_classes)``
          when ``num_classes`` is provided),
        * ``video_name`` set equals ``expected_video_names`` (when provided).

    Args:
        csv_path: Path to the submission CSV.
        num_classes: Optional. When set, predictions outside ``[0, num_classes)``
            cause a failure.
        expected_video_names: Optional iterable of names. When provided, the
            file's video names must match this set exactly.

    Returns:
        :class:`SubmissionReport` summarising the file.

    Raises:
        SubmissionFormatError: If any check fails.
        FileNotFoundError: If ``csv_path`` does not exist.
    """
    csv_path = Path(csv_path).resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Submission CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise SubmissionFormatError(f"{csv_path} is empty.") from exc

        if header != ["video_name", "predicted_class"]:
            raise SubmissionFormatError(
                f"{csv_path}: unexpected header {header!r}; "
                f"expected ['video_name', 'predicted_class']."
            )

        video_names: list[str] = []
        predictions: list[int] = []
        for row_idx, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise SubmissionFormatError(
                    f"{csv_path}: row {row_idx} has {len(row)} cells; expected 2."
                )
            name, raw_pred = row
            if not name:
                raise SubmissionFormatError(f"{csv_path}: row {row_idx} has an empty video_name.")
            try:
                pred = int(raw_pred)
            except ValueError as exc:
                raise SubmissionFormatError(
                    f"{csv_path}: row {row_idx} predicted_class={raw_pred!r} is not an integer."
                ) from exc
            if num_classes is not None and not 0 <= pred < num_classes:
                raise SubmissionFormatError(
                    f"{csv_path}: row {row_idx} predicted_class={pred} outside [0, {num_classes})."
                )
            video_names.append(name)
            predictions.append(pred)

    unique = set(video_names)
    if len(unique) != len(video_names):
        duplicates = [n for n in video_names if video_names.count(n) > 1]
        raise SubmissionFormatError(
            f"{csv_path}: {len(video_names) - len(unique)} duplicate video_name(s) "
            f"(e.g. {sorted(set(duplicates))[:3]})."
        )

    if expected_video_names is not None:
        expected = set(expected_video_names)
        missing = expected - unique
        extra = unique - expected
        if missing or extra:
            raise SubmissionFormatError(
                f"{csv_path}: video_name set mismatch. "
                f"missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}"
            )

    return SubmissionReport(
        path=csv_path,
        num_rows=len(video_names),
        unique_videos=len(unique),
        predictions=predictions,
        video_names=video_names,
    )


def write_submission_csv(
    output_path: Path,
    video_names: list[str],
    predictions: list[int],
) -> Path:
    """Write the submission CSV with header ``video_name,predicted_class``.

    Args:
        output_path: Destination CSV. Parent directories are created if missing.
        video_names: Ordered video names.
        predictions: Predicted class indices, one per video name.

    Returns:
        Resolved absolute path written.

    Raises:
        ValueError: If ``len(video_names) != len(predictions)``.
    """
    if len(video_names) != len(predictions):
        raise ValueError(
            f"video_names ({len(video_names)}) and predictions "
            f"({len(predictions)}) must have the same length."
        )

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_name", "predicted_class"])
        for name, pred in zip(video_names, predictions, strict=True):
            writer.writerow([name, int(pred)])
    return output_path
