"""Motion-aware time-reversal augmentation with class-pair remap.

For a *temporal* augmentation to be safe on an action-anticipation task, the
label must remain semantically correct under the augmentation. Horizontal
flip already has this property for all classes except the left/right pair
(handled by :mod:`smth2smth.pipelines.submit`'s flip permutation). Time
reversal is similar but the pairs are different:

* ``Opening_something`` <-> ``Closing_something``
* ``Putting_something_into_something`` <-> ``Taking_something_out_of_something``
* ``Folding_something`` <-> ``Unfolding_something``
* ``Covering_something_with_something`` <-> ``Uncovering_something``
* ``Moving_something_up`` <-> ``Moving_something_down``
* ``Moving_something_closer_to_something`` <-> ``Moving_something_away_from_something``
* ``Pouring_something_into_something`` <-> ``Pouring_something_out_of_something``

Classes that have no clean temporal mirror in the Track-A class set (e.g.
``Dropping_something_into_something``, ``Throwing_something``,
``Pretending_to_throw_something``) are **not** time-reversed: doing so would
corrupt the supervision signal. Symmetric classes whose label is unaffected
by reversal (``Holding_something`` for instance) keep the same index.

The helper :func:`build_time_reversal_table` walks the class folder names
once and returns a length-``num_classes`` ``LongTensor`` that maps each
class index to its post-reversal index, plus a boolean mask indicating
which classes are allowed to be reversed at all.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from smth2smth.shared.data.video_dataset import parse_class_index

# Canonical pair table keyed by the *class folder stem* (the leading
# ``\d+_`` prefix is stripped before matching, so ``003_Folding_something``
# pairs with ``032_Unfolding_something``).
_REVERSAL_PAIRS_BY_NAME: dict[str, str] = {
    "Opening_something": "Closing_something",
    "Closing_something": "Opening_something",
    "Putting_something_into_something": "Taking_something_out_of_something",
    "Taking_something_out_of_something": "Putting_something_into_something",
    "Folding_something": "Unfolding_something",
    "Unfolding_something": "Folding_something",
    "Covering_something_with_something": "Uncovering_something",
    "Uncovering_something": "Covering_something_with_something",
    "Moving_something_up": "Moving_something_down",
    "Moving_something_down": "Moving_something_up",
    "Moving_something_closer_to_something": "Moving_something_away_from_something",
    "Moving_something_away_from_something": "Moving_something_closer_to_something",
    "Pouring_something_into_something": "Pouring_something_out_of_something",
    "Pouring_something_out_of_something": "Pouring_something_into_something",
}

# Classes whose semantics do *not* change under temporal reversal. Reversal
# is allowed for these but the label stays the same.
_REVERSAL_SYMMETRIC: frozenset[str] = frozenset(
    {
        "Holding_something",
    }
)


def _strip_prefix(name: str) -> str:
    return re.sub(r"^\d+_", "", name)


def build_time_reversal_table(
    train_dir: Path,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the per-class reversal remap and allow-mask.

    Args:
        train_dir: Path to the class-bucketed training folder
            (``data/train/<class>/<video>/...``). Class indices and stems are
            read from the folder names.
        num_classes: Width of the classifier head.

    Returns:
        Tuple ``(perm, allow_mask)``:

        * ``perm``: ``LongTensor(num_classes,)``. ``perm[c]`` is the class
          index a frame should be labelled with after temporal reversal.
          For non-reversible classes, ``perm[c] == c`` (identity) **but**
          the corresponding allow-mask entry will be ``False``.
        * ``allow_mask``: ``BoolTensor(num_classes,)``. ``True`` means the
          class is part of a known reversible pair (or is in
          :data:`_REVERSAL_SYMMETRIC`); reversal is safe to apply to a
          sample of this class. ``False`` means leave the sample alone.

    Notes:
        The function never raises if a paired class is absent on disk; the
        corresponding entries simply stay at identity and are excluded from
        ``allow_mask`` -- partial coverage is still useful.
    """
    perm = list(range(num_classes))
    allow = [False] * num_classes

    if not train_dir.is_dir():
        return torch.tensor(perm, dtype=torch.long), torch.tensor(allow, dtype=torch.bool)

    class_dirs = sorted(p for p in train_dir.iterdir() if p.is_dir())
    stem_by_idx: dict[int, str] = {}
    for d in class_dirs:
        idx = parse_class_index(d.name)
        if idx is None or not (0 <= idx < num_classes):
            continue
        stem_by_idx[idx] = _strip_prefix(d.name)

    for idx, stem in stem_by_idx.items():
        if stem in _REVERSAL_SYMMETRIC:
            perm[idx] = idx
            allow[idx] = True
            continue
        mirror = _REVERSAL_PAIRS_BY_NAME.get(stem)
        if mirror is None:
            continue
        for other_idx, other_stem in stem_by_idx.items():
            if other_stem == mirror:
                perm[idx] = other_idx
                allow[idx] = True
                break

    return torch.tensor(perm, dtype=torch.long), torch.tensor(allow, dtype=torch.bool)


def describe_table(perm: torch.Tensor, allow_mask: torch.Tensor) -> str:
    """Return a short human-readable string for logging.

    The output lists every class index whose reversal target is *different*
    from itself; this is the practically interesting subset for diagnostics.
    """
    pairs: list[str] = []
    for c in range(perm.numel()):
        if bool(allow_mask[c]) and int(perm[c]) != c:
            pairs.append(f"{c}->{int(perm[c])}")
    n_allowed = int(allow_mask.sum().item())
    if not pairs:
        return f"time-reversal: {n_allowed} class(es) allowed, no class-pair remap detected."
    return (
        f"time-reversal: {n_allowed} class(es) allowed; "
        f"pair remaps = [{', '.join(pairs)}]."
    )


__all__ = [
    "build_time_reversal_table",
    "describe_table",
]
