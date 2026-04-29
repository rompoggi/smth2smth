#!/usr/bin/env python3
"""Track A preset: closed-world (from-scratch) training.

Forwards every CLI argument to ``smth2smth.pipelines.train`` after pinning
``track=a`` and the matching experiment. Override anything else as usual::

    python scripts/run_track_a.py training.epochs=20 training.batch_size=16
    python scripts/run_track_a.py -m training.lr=1e-3,5e-4 training.epochs=20
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

TRACK_OVERRIDES = ["track=a", "experiment=baseline_from_scratch"]

_VALUE_FLAGS: frozenset[str] = frozenset(
    {
        "--cfg",
        "--package",
        "--config-path",
        "--config-name",
        "--config-dir",
        "--experimental-rerun",
        "--info",
    }
)


def _partition_args(user_args: list[str]) -> tuple[list[str], list[str]]:
    """Split user-supplied argv into Hydra flags (with their values) and overrides.

    Hydra positional overrides are tokens that don't start with ``-`` (e.g.
    ``key=value``, ``+key=value``, ``~key``). Everything else is a flag or a
    value attached to a flag.
    """
    flags: list[str] = []
    overrides: list[str] = []
    i = 0
    while i < len(user_args):
        arg = user_args[i]
        if arg.startswith("-"):
            flags.append(arg)
            if "=" not in arg and arg in _VALUE_FLAGS and i + 1 < len(user_args):
                flags.append(user_args[i + 1])
                i += 2
                continue
        else:
            overrides.append(arg)
        i += 1
    return flags, overrides


def main() -> None:
    user_flags, user_overrides = _partition_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *user_flags, *TRACK_OVERRIDES, *user_overrides]

    os.chdir(REPO_ROOT)

    from smth2smth.pipelines.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
