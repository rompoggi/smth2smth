"""Tests for the TTA helpers in :mod:`smth2smth.pipelines.submit`.

The submission pipeline itself is exercised end-to-end by
``test_smoke_end_to_end.py``; these tests target the new TTA utilities in
isolation so a regression in the flip-pair remap or softmax averaging can be
caught quickly.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from smth2smth.pipelines.submit import (
    _build_flip_class_permutation,
    _predict,
    _videomae_logits_batch,
)
from smth2smth.shared.models import build_model


class _ConstLogitsModel(nn.Module):
    """Returns a fixed logits tensor regardless of input.

    When ``flipped_logits`` is provided, alternates between ``logits`` (call 1)
    and ``flipped_logits`` (call 2) and so on. This matches the order
    ``_predict`` calls the model in TTA mode (original first, flipped second
    per batch), which lets us assert exact predictions.
    """

    def __init__(self, logits: torch.Tensor, flipped_logits: torch.Tensor | None = None) -> None:
        super().__init__()
        self.logits = logits
        self.flipped_logits = flipped_logits
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.flipped_logits is not None and self.calls % 2 == 0:
            return self.flipped_logits.expand(x.shape[0], -1)
        return self.logits.expand(x.shape[0], -1)


class _OneSampleDataset(Dataset):
    def __init__(self, video: torch.Tensor) -> None:
        self.video = video

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _idx: int) -> tuple[torch.Tensor, int]:
        return self.video, 0


class TestVideomaePerceiverSubmitPath:
    """Regression: ViT TTA submit must route through pool_head, not mean-pool."""

    def test_videomae_logits_batch_matches_forward_for_perceiver(self) -> None:
        cfg = OmegaConf.create(
            {
                "model": {
                    "name": "video_mae_vit",
                    "variant": "vit_b",
                    "num_classes": 33,
                    "tube_t": 1,
                    "patch_size": 16,
                    "head": "perceiver",
                    "head_queries": 16,
                    "head_num_heads": 12,
                    "head_mlp_ratio": 4.0,
                    "drop_path_rate": 0.0,
                    "dropout": 0.0,
                    "gradient_checkpointing": False,
                    "hc_n": 4,
                },
                "dataset": {"num_frames": 4, "image_size": 224},
            }
        )
        model = build_model(cfg).eval()
        x = torch.randn(2, 4, 3, 224, 224)
        with torch.no_grad():
            ref = model(x)
            got = _videomae_logits_batch(model, x, untrained_mask=None, logit_adjust=None)
        assert torch.allclose(ref, got, atol=1e-4, rtol=1e-4)


class TestFlipClassPermutation:
    """Auto-derived left/right pairs from class folder names."""

    def test_returns_none_when_dir_missing(self) -> None:
        perm = _build_flip_class_permutation(
            train_dir=Path("/this/path/should/not/exist/xyz"),
            num_classes=5,
            device=torch.device("cpu"),
        )
        assert perm is None

    def test_pairs_left_right_classes(self, tmp_path: Path) -> None:
        for name in (
            "000_Closing",
            "018_Pulling_something_from_left_to_right",
            "019_Pulling_something_from_right_to_left",
            "032_Unfolding_something",
        ):
            (tmp_path / name).mkdir()
        perm = _build_flip_class_permutation(
            train_dir=tmp_path,
            num_classes=33,
            device=torch.device("cpu"),
        )
        assert perm is not None
        assert int(perm[18].item()) == 19
        assert int(perm[19].item()) == 18
        # All other classes map to themselves.
        for c in (0, 1, 17, 20, 32):
            assert int(perm[c].item()) == c


class TestPredict:
    """Argmax prediction with and without flip-TTA."""

    def test_no_tta_matches_argmax(self) -> None:
        logits = torch.tensor([[1.0, 5.0, 2.0]])
        model = _ConstLogitsModel(logits)
        loader = DataLoader(_OneSampleDataset(torch.randn(2, 3, 8, 8)), batch_size=1)
        preds = _predict(
            model=model,
            loader=loader,
            device=torch.device("cpu"),
            untrained_mask=None,
            tta_enabled=False,
            tta_flip=False,
            flip_perm=None,
        )
        assert preds == [1]

    def test_untrained_mask_blocks_class(self) -> None:
        logits = torch.tensor([[1.0, 5.0, 2.0]])
        model = _ConstLogitsModel(logits)
        loader = DataLoader(_OneSampleDataset(torch.randn(2, 3, 8, 8)), batch_size=1)
        # Mask class 1 (the natural argmax) -> argmax becomes class 2.
        mask = torch.tensor([0.0, float("-inf"), 0.0])
        preds = _predict(
            model=model,
            loader=loader,
            device=torch.device("cpu"),
            untrained_mask=mask,
            tta_enabled=False,
            tta_flip=False,
            flip_perm=None,
        )
        assert preds == [2]

    def test_flip_tta_with_remap_aligns_classes(self) -> None:
        # Original frame: argmax=1 (very confident).
        # Flipped frame: argmax=2 (very confident).
        # Without remap, the average has near-equal mass at 1 and 2 (tie-break
        # picks the lowest index, 1). With the {1,2} swap remap, the flipped
        # softmax peak at class 2 is rotated to class 1, so the average
        # strongly reinforces class 1.
        logits = torch.tensor([[0.0, 10.0, 0.0]])
        flipped = torch.tensor([[0.0, 0.0, 10.0]])
        clip = torch.linspace(0.0, 1.0, 8).expand(2, 3, 8, 8).contiguous()

        # Identity remap (no swap): tie-broken to class 1.
        model = _ConstLogitsModel(logits, flipped_logits=flipped)
        loader = DataLoader(_OneSampleDataset(clip), batch_size=1)
        preds = _predict(
            model=model,
            loader=loader,
            device=torch.device("cpu"),
            untrained_mask=None,
            tta_enabled=True,
            tta_flip=True,
            flip_perm=torch.tensor([0, 1, 2], dtype=torch.long),
        )
        assert preds == [1]

        # Swap-perm: the flipped softmax peak (class 2) is remapped to class 1
        # before averaging -- argmax remains class 1 unambiguously.
        model = _ConstLogitsModel(logits, flipped_logits=flipped)
        loader = DataLoader(_OneSampleDataset(clip), batch_size=1)
        preds = _predict(
            model=model,
            loader=loader,
            device=torch.device("cpu"),
            untrained_mask=None,
            tta_enabled=True,
            tta_flip=True,
            flip_perm=torch.tensor([0, 2, 1], dtype=torch.long),
        )
        assert preds == [1]
