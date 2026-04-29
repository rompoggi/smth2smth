"""Classification metrics (top-k accuracy)."""

from __future__ import annotations

import torch


@torch.no_grad()
def accuracy_topk(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: tuple[int, ...] = (1, 5),
) -> tuple[torch.Tensor, ...]:
    """Compute top-k accuracy for each ``k`` in ``topk``.

    Args:
        logits: Tensor of shape ``(batch_size, num_classes)``.
        targets: Tensor of shape ``(batch_size,)`` with integer class indices.
        topk: Tuple of k values to evaluate.

    Returns:
        Tuple of tensors, each shape ``(1,)`` with accuracy in ``[0, 1]``.
        Order matches the order of ``topk``.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2-D (batch, classes); got shape {tuple(logits.shape)}")
    if targets.dim() != 1:
        raise ValueError(f"targets must be 1-D (batch,); got shape {tuple(targets.shape)}")

    num_classes = logits.size(1)
    batch_size = targets.size(0)
    if batch_size == 0:
        zero = torch.zeros(1, device=logits.device)
        return tuple(zero for _ in topk)

    # Clamp k to the number of classes (e.g. top-5 on a 3-class problem trivially equals 1.0).
    effective_topk = tuple(min(k, num_classes) for k in topk)
    max_k = max(effective_topk)

    _, predictions = logits.topk(max_k, dim=1, largest=True, sorted=True)
    predictions = predictions.t()
    correct = predictions.eq(targets.view(1, -1).expand_as(predictions))

    return tuple(correct[:k].reshape(-1).float().sum() / batch_size for k in effective_topk)
