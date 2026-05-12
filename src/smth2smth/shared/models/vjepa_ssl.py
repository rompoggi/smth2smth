"""V-JEPA-style clip-level self-supervised pretraining.

This module provides a *Track-A-compliant* alternative to the still-frame DINO
pipeline in :mod:`smth2smth.shared.models.dino_ssl`. Its central observation
is that DINO on isolated frames teaches the trunk to be **invariant** to
spatial augmentation -- exactly the wrong inductive bias for a TSM-based
action-anticipation model, which needs to *exploit* small temporal changes
between frames.

The fix follows the V-JEPA 2 recipe (Assran et al. 2025), specialised here
to a ResNet-50 + Temporal Shift Module trunk:

1. We operate on full ``T``-frame **clips**, not still frames, so the
   :class:`smth2smth.shared.models.avanced_resnet50_tsm.TemporalShift`
   channel-shift modules are exercised during SSL.
2. We **mask** a random subset of frames in each clip (replacing their
   normalised tensors with zeros). The student trunk sees the masked clip;
   the teacher trunk sees the unmasked clip with ``no_grad``.
3. A small **predictor** maps the student's per-frame feature back to the
   teacher's space, where an **L1 loss is applied on masked positions
   only** -- the V-JEPA mask-denoising objective.

Because the trunk *is* the supervised model's backbone (same TSM wrapping,
same ResNet-50 layout), the resulting checkpoint loads directly via the
existing ``model.init_from`` hook in :mod:`smth2smth.pipelines.train`
(the helper :func:`_ssl_trunk_to_supervised_keys` is a no-op when the
trunk keys are already TSM-wrapped, and just prefixes with ``backbone.``).

Compared to the DINO trunk in :mod:`smth2smth.shared.models.dino_ssl`:

* Uses the **TSM-wrapped backbone** instead of a plain ResNet-50. This is
  the structural fix to "SSL pretraining sometimes destroys accuracy" --
  the TemporalShift channels are now part of the pretraining loss.
* Predicts in **feature space** rather than via prototype clustering. The
  L1 objective makes training stable on small datasets and does not need
  a centring/sharpening machinery.
* Trains on **clips** (so the dataloader iterates videos, not frames),
  which keeps the supervised and SSL data pipelines symmetric.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from smth2smth.shared.models.avanced_resnet50_tsm import (
    _make_temporal_shift_resnet,
    _validate_residual_paths,
)


class VJepaTrunk(nn.Module):
    """TSM-equipped ResNet-50 trunk usable as a video encoder for V-JEPA SSL.

    State-dict layout is **identical** to
    :class:`smth2smth.shared.models.avanced_resnet50_tsm.AvancedResNet50TSM`'s
    ``backbone`` attribute, so the trained weights load straight into the
    supervised model via the existing ``model.init_from`` hook.

    Forward contract:
        * Input ``(B, T, C, H, W)``  ->  per-frame features ``(B, T, D)``
          where ``D = 2048`` (ResNet-50 last-stage channels).
        * Internally reshapes to ``(B * T, C, H, W)`` so the inner
          ``TemporalShift`` modules see the expected first-dim contract
          (i.e. ``n_segment == T``).
    """

    def __init__(self, num_frames: int, shift_div: int = 8) -> None:
        super().__init__()
        if num_frames < 2:
            raise ValueError(
                f"num_frames must be >= 2 for V-JEPA SSL with TSM, got {num_frames}."
            )
        backbone = models.resnet50(weights=None)
        self.feature_dim = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        backbone = _make_temporal_shift_resnet(
            backbone=backbone,
            n_segment=num_frames,
            shift_div=shift_div,
            shift_place="blockres",
        )
        _validate_residual_paths(backbone)
        self.backbone = backbone
        self.num_frames = int(num_frames)
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming He fan-out init for ReLU convs, BN=1/0 (matches the
        supervised AvancedResNet50TSM recipe so the SSL run starts from the
        same distribution as a plain from-scratch supervised run)."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """``clip``: ``(B, T, C, H, W)`` -> ``(B, T, D)``."""
        if clip.dim() != 5:
            raise ValueError(f"VJepaTrunk expects (B,T,C,H,W); got {tuple(clip.shape)}")
        b, t, c, h, w = clip.shape
        if t != self.num_frames:
            raise ValueError(
                f"VJepaTrunk was built for num_frames={self.num_frames} but got T={t}."
            )
        feats = self.backbone(clip.reshape(b * t, c, h, w))
        feats = torch.flatten(feats, start_dim=1)  # (B*T, D)
        return feats.view(b, t, -1)


class VJepaPredictor(nn.Module):
    """Tiny per-frame MLP that maps student features to teacher space.

    V-JEPA's predictor exists so the encoder isn't biased toward an identity
    student-to-teacher mapping at the masked positions. With a CNN trunk the
    natural form is a small per-frame MLP applied along the feature axis;
    we use ``Linear -> GELU -> Linear`` with hidden dim equal to the input
    dim (Bardes et al. 2024, Sec. 3 -- minimal predictor recipe).

    Applied independently to each of the ``T`` frame features. Output shape
    matches the input.
    """

    def __init__(self, feature_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        h = int(hidden_dim or feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, h),
            nn.GELU(),
            nn.Linear(h, feature_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        """``frame_features``: ``(B, T, D)`` -> ``(B, T, D)``."""
        return self.mlp(frame_features)


class VJepaModel(nn.Module):
    """Composite student model: TSM trunk + V-JEPA predictor.

    The trunk produces per-frame ``D``-dim features. The predictor maps them
    to the teacher's feature space. At training time, both run for every
    batch; at the end of pretraining we save only ``trunk.state_dict()``
    (with ``trunk.`` stripped) so the supervised pipeline picks the weights
    up via ``model.init_from`` without modification.

    The teacher is **not** part of this module -- it is held externally by
    :mod:`smth2smth.pipelines.pretrain_vjepa` as a deep copy with EMA-only
    updates, mirroring the existing DINO recipe in
    :mod:`smth2smth.pipelines.pretrain_ssl`.
    """

    def __init__(
        self,
        num_frames: int,
        shift_div: int = 8,
        predictor_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.trunk = VJepaTrunk(num_frames=num_frames, shift_div=shift_div)
        self.predictor = VJepaPredictor(
            feature_dim=self.trunk.feature_dim,
            hidden_dim=predictor_hidden_dim,
        )

    def encode(self, clip: torch.Tensor) -> torch.Tensor:
        """Trunk-only forward: ``(B, T, C, H, W)`` -> ``(B, T, D)``.

        Used for the **teacher** path during training (no predictor, no
        gradient -- the caller is responsible for ``torch.no_grad()``).
        """
        return self.trunk(clip)

    def forward(self, clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Student path. Returns ``(predicted_features, raw_features)``.

        ``predicted_features`` is what the V-JEPA loss compares to the
        teacher's encoded features at masked positions. ``raw_features``
        is the trunk's pre-predictor output, exposed for diagnostics
        (e.g. monitoring representational variance).
        """
        feats = self.trunk(clip)  # (B, T, D)
        predicted = self.predictor(feats)
        return predicted, feats


def make_frame_mask(
    batch_size: int,
    num_frames: int,
    mask_prob: float,
    min_mask: int = 1,
    max_mask: int | None = None,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a ``(B, T)`` boolean frame mask (``True`` = masked).

    For each clip independently:
        * sample ``n_masked ~ Uniform[min_mask, max_mask]``;
        * pick ``n_masked`` frame indices uniformly without replacement;
        * mark them ``True``.

    The default ``mask_prob`` (used to scale ``max_mask`` when caller passes
    ``None``) is wired so that the average mask fraction roughly matches the
    requested probability while never producing degenerate ``all-masked`` or
    ``no-masked`` clips: we clip ``max_mask`` to ``num_frames - 1`` (the
    student must always have at least one unmasked frame to propagate
    through the TSM channels).

    Args:
        batch_size: Number of clips ``B``.
        num_frames: Frames per clip ``T``.
        mask_prob: Approximate per-clip mask fraction in ``[0, 1)``. Used
            to derive ``max_mask`` when the caller doesn't override it.
        min_mask: Minimum number of frames to mask per clip (``>= 1``).
        max_mask: Hard upper bound on ``n_masked``; defaults to
            ``round(mask_prob * num_frames)`` capped at ``num_frames - 1``.
        device: Device for the returned mask.
        generator: Optional pre-seeded ``torch.Generator`` for determinism.

    Returns:
        ``BoolTensor`` of shape ``(B, T)`` with ``True`` at masked positions.

    Raises:
        ValueError: On out-of-range arguments.
    """
    if num_frames < 2:
        raise ValueError(f"num_frames must be >= 2, got {num_frames}.")
    if not 0.0 <= mask_prob < 1.0:
        raise ValueError(f"mask_prob must be in [0, 1), got {mask_prob}.")
    if min_mask < 1:
        raise ValueError(f"min_mask must be >= 1, got {min_mask}.")
    if max_mask is None:
        max_mask = max(min_mask, int(round(mask_prob * num_frames)))
    max_mask = max(min_mask, min(int(max_mask), num_frames - 1))
    if min_mask > max_mask:
        raise ValueError(
            f"min_mask={min_mask} > max_mask={max_mask}; cannot sample a non-degenerate mask."
        )

    mask = torch.zeros(batch_size, num_frames, dtype=torch.bool, device=device)
    for b in range(batch_size):
        n = int(
            torch.randint(
                low=min_mask, high=max_mask + 1, size=(1,), device=device, generator=generator
            ).item()
        )
        perm = torch.randperm(num_frames, device=device, generator=generator)
        mask[b, perm[:n]] = True
    return mask


def apply_frame_mask(clip: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero out masked frames in a ``(B, T, C, H, W)`` clip.

    Args:
        clip: ``(B, T, C, H, W)`` tensor.
        mask: ``(B, T)`` boolean tensor; ``True`` at positions to zero.

    Returns:
        A new tensor with the masked frames replaced by zeros.
    """
    if clip.dim() != 5:
        raise ValueError(f"clip must be 5-D (B,T,C,H,W), got {tuple(clip.shape)}")
    if mask.shape != (clip.shape[0], clip.shape[1]):
        raise ValueError(
            f"mask shape {tuple(mask.shape)} != (B={clip.shape[0]}, T={clip.shape[1]})"
        )
    keep = (~mask).to(dtype=clip.dtype).view(*mask.shape, 1, 1, 1)
    return clip * keep


def vjepa_feature_loss(
    predicted: torch.Tensor,
    teacher: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """V-JEPA L1 feature-prediction loss, masked-positions only.

    The teacher tensor must be already detached (no grad). The loss is
    averaged over masked frame positions and feature dimensions: this is
    the canonical V-JEPA objective (Bardes et al. 2024, eq. 1).

    Numerics: the loss arithmetic is performed in **float32** even when
    the caller passes fp16 tensors (e.g. under ``torch.amp.autocast``).
    This is required because the per-batch denominator ``mask.sum() * D``
    can exceed fp16's max representable value (65504) for reasonable
    batch sizes -- e.g. with ``B=32``, ``T=4``, ``D=2048`` and one frame
    masked per clip, the denominator is already ``32 * 2048 = 65536``.
    A naive fp16 division silently saturates to ``+inf`` and the
    resulting loss is identically zero (with no gradient signal).
    Computing in fp32 here is the cheap fix that keeps fp16 throughput
    in the forward pass while preventing this silent failure.

    Args:
        predicted: Student-predictor output, ``(B, T, D)``.
        teacher: Teacher trunk output, ``(B, T, D)``, ``.detach()``ed.
        mask: ``(B, T)`` boolean; ``True`` ⇒ frame is masked and contributes
            to the loss. If the mask is empty, returns ``0`` (no learning
            signal this step).
        reduction: ``"mean"`` (default) or ``"sum"``.

    Returns:
        Scalar **float32** loss tensor on the same device as ``predicted``.

    Raises:
        ValueError: On shape mismatch or unknown reduction.
    """
    if predicted.shape != teacher.shape:
        raise ValueError(
            f"predicted {tuple(predicted.shape)} != teacher {tuple(teacher.shape)}"
        )
    if mask.shape != predicted.shape[:2]:
        raise ValueError(
            f"mask {tuple(mask.shape)} incompatible with features "
            f"{tuple(predicted.shape[:2])}"
        )
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}.")

    if not mask.any():
        # No masked frames in this batch; return a zero loss that still
        # back-propagates through ``predicted`` to keep the autograd graph
        # well-defined (avoids "loss has no grad_fn" failures downstream).
        return predicted.float().sum() * 0.0

    # Upcast to fp32 for the loss math (see numerics note in the docstring).
    predicted_f32 = predicted.float()
    teacher_f32 = teacher.float()
    diff = (predicted_f32 - teacher_f32).abs()  # (B, T, D) float32
    sel = mask.to(diff.dtype).unsqueeze(-1)  # (B, T, 1) float32
    weighted = diff * sel
    if reduction == "sum":
        return weighted.sum()
    # mean over masked positions and feature dims: total elements that
    # contributed is ``mask.sum() * D``. Kept in fp32 to avoid overflow.
    n_elems = mask.sum().clamp_min(1).to(torch.float32) * float(diff.shape[-1])
    return weighted.sum() / n_elems


@torch.no_grad()
def update_vjepa_teacher_ema(
    student: nn.Module,
    teacher: nn.Module,
    momentum: float,
) -> None:
    """In-place EMA update ``teacher = m * teacher + (1 - m) * student``.

    Mirrors :func:`smth2smth.shared.models.dino_ssl.update_teacher_ema`. We
    keep a dedicated entry point so the V-JEPA pipeline doesn't need to
    import from the DINO module (and so future divergences -- e.g. updating
    only the trunk and not the predictor -- can be localised here).
    """
    for p_s, p_t in zip(student.parameters(), teacher.parameters(), strict=True):
        p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)


__all__ = [
    "VJepaModel",
    "VJepaPredictor",
    "VJepaTrunk",
    "apply_frame_mask",
    "make_frame_mask",
    "update_vjepa_teacher_ema",
    "vjepa_feature_loss",
]
