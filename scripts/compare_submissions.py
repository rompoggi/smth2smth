#!/usr/bin/env python3
"""Compare two submission CSVs by ``video_name``.

Both files must have header ``video_name,predicted_class`` and the same set
of video names (the order can differ — comparison is keyed on ``video_name``).

Typical use::

    python scripts/compare_submissions.py \\
        --baseline external/prof_baseline/submission.csv \\
        --ours submissions/track_b.csv

Prints a small report:
    * number of rows,
    * agreement rate (% of identical predictions),
    * top-K disagreement classes (where they predict different classes the most),
    * the first few disagreement rows.

Exit code 0 when both files are valid and comparable, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smth2smth.shared.io.submission import (  # noqa: E402  (sys.path set above)
    SubmissionFormatError,
    validate_submission_csv,
)


@dataclass
class ComparisonResult:
    """Summary of comparing two submission CSVs.

    Attributes:
        num_videos: Number of videos shared by both files.
        agreement: Fraction of videos with identical predictions in ``[0, 1]``.
        agree_count: Count of identical predictions.
        disagree_count: Count of differing predictions.
        top_disagreement_pairs: List of ``((cls_a, cls_b), count)`` for the most
            common disagreement directions.
    """

    num_videos: int
    agreement: float
    agree_count: int
    disagree_count: int
    top_disagreement_pairs: list[tuple[tuple[int, int], int]]


def compare(
    baseline_predictions: dict[str, int],
    ours_predictions: dict[str, int],
    *,
    top_k: int = 5,
) -> ComparisonResult:
    """Compare two ``video_name -> predicted_class`` mappings.

    Args:
        baseline_predictions: Mapping from baseline submission.
        ours_predictions: Mapping from our submission.
        top_k: How many top-disagreement (class_a, class_b) pairs to report.

    Returns:
        :class:`ComparisonResult`.

    Raises:
        ValueError: When the two mappings have different sets of video names.
    """
    baseline_keys = set(baseline_predictions)
    ours_keys = set(ours_predictions)
    if baseline_keys != ours_keys:
        only_baseline = baseline_keys - ours_keys
        only_ours = ours_keys - baseline_keys
        raise ValueError(
            f"video_name mismatch: only_baseline={sorted(only_baseline)[:5]} "
            f"only_ours={sorted(only_ours)[:5]}"
        )

    pair_counts: Counter[tuple[int, int]] = Counter()
    agree_count = 0
    for video_name in baseline_keys:
        a = baseline_predictions[video_name]
        b = ours_predictions[video_name]
        if a == b:
            agree_count += 1
        else:
            pair_counts[(a, b)] += 1

    num_videos = len(baseline_keys)
    disagree_count = num_videos - agree_count
    agreement = agree_count / num_videos if num_videos else 0.0
    return ComparisonResult(
        num_videos=num_videos,
        agreement=agreement,
        agree_count=agree_count,
        disagree_count=disagree_count,
        top_disagreement_pairs=pair_counts.most_common(top_k),
    )


def _print_report(baseline_path: Path, ours_path: Path, result: ComparisonResult) -> None:
    print(f"baseline : {baseline_path}")
    print(f"ours     : {ours_path}")
    print(f"videos   : {result.num_videos}")
    print(f"agree    : {result.agree_count}  ({result.agreement * 100:.2f}%)")
    print(f"disagree : {result.disagree_count}")
    if result.top_disagreement_pairs:
        print("top disagreement directions (baseline_class -> ours_class):")
        for (cls_a, cls_b), count in result.top_disagreement_pairs:
            print(f"  {cls_a:>3} -> {cls_b:<3}  {count}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", required=True, type=Path, help="Baseline CSV")
    parser.add_argument("--ours", required=True, type=Path, help="Our submission CSV")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Optional. When set, predictions outside [0, num_classes) cause failure.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top-disagreement (class_a, class_b) pairs to print.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        baseline_report = validate_submission_csv(args.baseline, num_classes=args.num_classes)
        ours_report = validate_submission_csv(args.ours, num_classes=args.num_classes)
    except (SubmissionFormatError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    baseline_map = dict(zip(baseline_report.video_names, baseline_report.predictions, strict=True))
    ours_map = dict(zip(ours_report.video_names, ours_report.predictions, strict=True))

    try:
        result = compare(baseline_map, ours_map, top_k=args.top_k)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _print_report(args.baseline, args.ours, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
