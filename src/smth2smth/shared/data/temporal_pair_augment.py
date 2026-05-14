"""Temporal reversal + opposite-class label swap for Track A training.

Only applies to a **curated** set of Something-Something-style pairs where
playing the clip backwards often matches the **paired** verb (label swap).
Used **on the training split only**; validation must stay unmodified.

Excluded pairs (not semantically reliable under time reversal + swap):
``002``/``028``, ``012``/``013``, ``020``/``021``; ``030`` is intentionally omitted.
"""

from __future__ import annotations

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
