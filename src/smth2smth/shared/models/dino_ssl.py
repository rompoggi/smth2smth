"""DINO-v1 self-supervised wrapper around a ResNet-50 trunk.

Reference: Caron et al. *"Emerging Properties in Self-Supervised Vision
Transformers"* (ICCV 2021). DINO is a teacher-student knowledge-distillation
framework where:

* the student and the teacher have identical architectures;
* the teacher's weights are an EMA of the student's;
* the student processes ALL views (2 globals + ``L`` locals);
* the teacher processes only the 2 global views;
* the loss is a softmax-cross-entropy over learned prototypes, with the
  teacher's logits shifted by an EMA centering term and sharpened with a
  small temperature.

This implementation is *trunk-only* on purpose: we keep the per-frame
ResNet-50 stack used by :class:`AvancedResNet50TSM` so the SSL-trained
weights can be loaded directly into the supervised model's ``backbone``.

Track-A compliance: SSL pretraining only consumes the *provided* frames
(train+val+test). It uses no external dataset and no pretrained weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DinoHead(nn.Module):
    """Projection MLP + L2-normalised weight-norm linear classifier.

    Mirrors the head in DINO's reference implementation:
    Linear → GELU → Linear → GELU → Linear → L2 → WeightNorm(Linear, K).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}.")
        layers: list[nn.Module] = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        # Weight-normalised final layer: keeps output norm decoupled from
        # direction, which DINO found stabilises training.
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


def build_resnet50_trunk() -> tuple[nn.Module, int]:
    """Return ``(trunk, feature_dim)`` matching :class:`AvancedResNet50TSM`'s
    ``backbone`` attribute. The classification head is replaced by ``Identity``
    so the trunk emits ``(B, 2048)`` features.
    """
    backbone = models.resnet50(weights=None)
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feature_dim


class DinoModel(nn.Module):
    """Trunk + DINO projection head, ready for student/teacher use."""

    def __init__(
        self,
        out_dim: int = 4096,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.trunk, feature_dim = build_resnet50_trunk()
        self.head = DinoHead(
            in_dim=feature_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            n_layers=n_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.trunk(x)
        return self.head(feats)


class DinoLoss(nn.Module):
    """Centered + sharpened cross-entropy between teacher and student logits.

    The teacher's logits are centred by an EMA of past batches (Caron et al.
    Algorithm 1) and sharpened with a smaller temperature than the student's.
    Only off-diagonal (teacher_view_i, student_view_j) pairs with ``i != j``
    contribute to the loss to avoid the trivial identity solution.
    """

    def __init__(
        self,
        out_dim: int,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.teacher_temp = float(teacher_temp)
        self.student_temp = float(student_temp)
        self.center_momentum = float(center_momentum)
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(
        self,
        student_logits_per_view: list[torch.Tensor],
        teacher_logits_per_global: list[torch.Tensor],
    ) -> torch.Tensor:
        # Sharpen + center the teacher (no grad).
        teacher_probs_per_global = [
            F.softmax((logits - self.center) / self.teacher_temp, dim=-1).detach()
            for logits in teacher_logits_per_global
        ]
        student_log_softmax_per_view = [
            F.log_softmax(logits / self.student_temp, dim=-1)
            for logits in student_logits_per_view
        ]
        total_loss = 0.0
        n_terms = 0
        for t_idx, t_probs in enumerate(teacher_probs_per_global):
            for s_idx, s_log in enumerate(student_log_softmax_per_view):
                if t_idx == s_idx:
                    # Skip same-view pairs (the canonical DINO recipe).
                    continue
                loss = -(t_probs * s_log).sum(dim=-1).mean()
                total_loss = total_loss + loss
                n_terms += 1
        if n_terms == 0:
            raise RuntimeError(
                "DinoLoss received no off-diagonal pairs; "
                "are there at least two student views?"
            )
        loss = total_loss / float(n_terms)

        # Update centering buffer (EMA of teacher logits across all globals).
        with torch.no_grad():
            stacked = torch.cat(teacher_logits_per_global, dim=0)
            batch_center = stacked.mean(dim=0, keepdim=True)
            self.center = (
                self.center_momentum * self.center + (1.0 - self.center_momentum) * batch_center
            )
        return loss


@torch.no_grad()
def update_teacher_ema(
    student: nn.Module,
    teacher: nn.Module,
    momentum: float,
) -> None:
    """In-place EMA update: ``teacher = m * teacher + (1 - m) * student``."""
    for p_s, p_t in zip(student.parameters(), teacher.parameters(), strict=True):
        p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)
