"""Generic utilities: seeding and dataset splits."""

from smth2smth.shared.utils.seed import set_seed
from smth2smth.shared.utils.splits import VideoSample, split_train_val

__all__ = ["VideoSample", "set_seed", "split_train_val"]
