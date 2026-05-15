"""Temporal reversal + opposite-class label swap for Track A training.

Only applies to a **curated** set of Something-Something-style pairs where
playing the clip backwards often matches the **paired** verb (label swap).
Used **on the training split only**; validation must stay unmodified.

Excluded pairs (not semantically reliable under time reversal + swap):
``002``/``028``, ``012``/``013``, ``020``/``021``; ``030`` is intentionally omitted.

**Class boosting** (deterministic): duplicate each training row whose disk label
lies in a curated pair: one row keeps the disk label and frame order, a second
row uses the **partner** label and **reversed** sampled frames. Implemented by
:func:`expand_train_samples_for_class_boosting` plus a ``(path, label, bool)``
flag consumed by :class:`VideoFrameDataset`.
"""

from __future__ import annotations

from pathlib import Path

# Undirected pairs as (class_index_a, class_index_b) using folder numeric prefixes.
TRACK_A_TEMPORAL_REVERSAL_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 10),  # Closing / Opening
    (1, 31),  # Covering / Uncovering
    (3, 32),  # Folding / Unfolding
    (6, 7),  # Away from / Closer to
    (8, 9),  # Down / Up
    (18, 19),  # Pull L→R / Pull R→L
    (22, 28),  # Put into / Take out of
)


def build_track_a_temporal_reversal_map() -> dict[int, int]:
    """Return a symmetric ``label -> opposite_label`` map for :class:`VideoFrameDataset`.

    Returns:
        Dict with ``d[a] == b`` and ``d[b] == a`` for every curated pair.
    """
    out: dict[int, int] = {}
    for a, b in TRACK_A_TEMPORAL_REVERSAL_PAIRS:
        out[int(a)] = int(b)
        out[int(b)] = int(a)
    return out


def expand_train_samples_for_class_boosting(
    samples: list[tuple[Path, int]],
    pair_to_opposite: dict[int, int],
) -> list[tuple[Path, int] | tuple[Path, int, bool]]:
    """Duplicate paired-class clips for deterministic **class boosting**.

    For each ``(video_dir, label)`` with ``label`` in ``pair_to_opposite``,
    appends two records:

    * ``(video_dir, label, False)`` — native order, disk label;
    * ``(video_dir, pair_to_opposite[label], True)`` — time-reversed frames,
      partner label.

    Unpaired classes stay as ``(video_dir, label)``.

    Args:
        samples: Training split after ``split_train_val`` (2-tuples only).
        pair_to_opposite: Symmetric map from :func:`build_track_a_temporal_reversal_map`.

    Returns:
        Expanded list; length is ``len(samples) +`` number of paired clips.
    """
    out: list[tuple[Path, int] | tuple[Path, int, bool]] = []
    for video_dir, label in samples:
        lab = int(label)
        if lab in pair_to_opposite:
            out.append((video_dir, lab, False))
            out.append((video_dir, int(pair_to_opposite[lab]), True))
        else:
            out.append((video_dir, lab))
    return out
