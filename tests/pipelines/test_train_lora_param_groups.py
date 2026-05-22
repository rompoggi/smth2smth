"""Tests for LoRA vs head parameter splitting in :mod:`smth2smth.pipelines.train`."""

from __future__ import annotations

import torch
import torch.nn as nn

from smth2smth.pipelines.train import _split_trainable_params_head_vs_lora


class _ToyWithLoraNames(nn.Module):
    """Minimal module using PEFT-style parameter names."""

    def __init__(self) -> None:
        super().__init__()
        self.head_w = nn.Parameter(torch.zeros(2, 2))
        self.backbone_blocks_0_attn_q_proj_lora_A_default = nn.Parameter(torch.zeros(1))
        self.backbone_blocks_0_attn_q_proj_lora_B_default = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return x


def test_split_trainable_head_vs_lora() -> None:
    model = _ToyWithLoraNames()
    head, lora = _split_trainable_params_head_vs_lora(model)
    assert len(head) == 1 and head[0] is model.head_w
    assert len(lora) == 2
    assert {id(p) for p in lora} == {
        id(model.backbone_blocks_0_attn_q_proj_lora_A_default),
        id(model.backbone_blocks_0_attn_q_proj_lora_B_default),
    }


def test_split_trainable_respects_requires_grad() -> None:
    model = _ToyWithLoraNames()
    model.head_w.requires_grad_(False)
    head, lora = _split_trainable_params_head_vs_lora(model)
    assert len(head) == 0
    assert len(lora) == 2


class _ToyWithDoraNames(nn.Module):
    """PEFT DoRA adds a ``lora_magnitude_vector`` alongside ``lora_A``/``lora_B``."""

    def __init__(self) -> None:
        super().__init__()
        self.head_w = nn.Parameter(torch.zeros(2, 2))
        self.attn_q_proj_lora_A_default = nn.Parameter(torch.zeros(1))
        self.attn_q_proj_lora_B_default = nn.Parameter(torch.zeros(1))
        self.attn_q_proj_lora_magnitude_vector_default = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return x


def test_dora_magnitude_vector_routed_to_lora_group() -> None:
    model = _ToyWithDoraNames()
    head, lora = _split_trainable_params_head_vs_lora(model)
    assert len(head) == 1 and head[0] is model.head_w
    # A/B plus the DoRA magnitude vector all get the dedicated LoRA LR.
    assert len(lora) == 3
    assert id(model.attn_q_proj_lora_magnitude_vector_default) in {id(p) for p in lora}
