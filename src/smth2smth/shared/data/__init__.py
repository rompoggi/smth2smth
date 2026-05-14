"""Shared data layer: dataset and image transforms."""

from smth2smth.shared.data.clip_ssl_dataset import ClipSSLDataset, collect_all_video_dirs
from smth2smth.shared.data.randaugment import RA_OPS, RandAugment
from smth2smth.shared.data.still_frames_dataset import (
    MultiViewStillFramesDataset,
    collect_all_frame_paths,
)
from smth2smth.shared.data.temporal_pair_augment import (
    TRACK_A_TEMPORAL_REVERSAL_PAIRS,
    build_track_a_temporal_reversal_map,
)
from smth2smth.shared.data.time_reversal import (
    build_time_reversal_table,
)
from smth2smth.shared.data.time_reversal import (
    describe_table as describe_time_reversal_table,
)
from smth2smth.shared.data.transforms import build_transforms
from smth2smth.shared.data.video_dataset import (
    VideoFrameDataset,
    VideoSample,
    collect_video_samples,
    parse_class_index,
    pick_frame_indices,
)

__all__ = [
    "RA_OPS",
    "ClipSSLDataset",
    "TRACK_A_TEMPORAL_REVERSAL_PAIRS",
    "MultiViewStillFramesDataset",
    "RandAugment",
    "VideoFrameDataset",
    "VideoSample",
    "build_time_reversal_table",
    "build_track_a_temporal_reversal_map",
    "build_transforms",
    "collect_all_frame_paths",
    "collect_all_video_dirs",
    "collect_video_samples",
    "describe_time_reversal_table",
    "parse_class_index",
    "pick_frame_indices",
]
