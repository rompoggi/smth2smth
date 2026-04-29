"""Seed helpers for reproducibility."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) RNGs.

    Args:
        seed: Non-negative integer used for all RNGs.

    Notes:
        Full determinism on CUDA still requires
        ``torch.use_deterministic_algorithms(True)`` and disabling cuDNN's
        autotuning, which we don't enforce here to keep training speed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
