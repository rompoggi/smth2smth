"""Tests for active augmentation summary logging."""

from __future__ import annotations

from smth2smth.shared.utils.augment_log import active_augment_summary


def test_only_enabled_augments() -> None:
    summary = active_augment_summary(
        {
            "name": "test",
            "random_crop": True,
            "crop_padding": 32,
            "random_horizontal_flip": False,
            "color_jitter": False,
            "randaugment": {"enabled": False, "n": 2},
        }
    )
    assert summary["augment_name"] == "test"
    assert summary["random_crop"] is True
    assert summary["crop_padding"] == 32
    assert "random_horizontal_flip" not in summary
    assert "color_jitter" not in summary
    assert "randaugment" not in summary
