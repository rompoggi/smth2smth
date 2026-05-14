"""V-JEPA 2 frozen-backbone video classifier for Track B.

Track B (Open World) allows pretrained backbones and external data, so this
is the highest-impact lever: V-JEPA 2 (Assran et al., FAIR 2025) is a
state-of-the-art self-supervised video encoder that reports **77.3 % top-1
on Something-Something v2** with an attentive probe on top of a frozen
ViT-g/16 384 encoder -- the same dataset family our 33-class subset is
drawn from. V-JEPA 2.1 (Mur-Labadia et al. 2026) further improves dense
feature quality via Dense Predictive Loss and Deep Self-Supervision.

This module wraps the encoder portion of a V-JEPA 2 checkpoint (loaded from
HuggingFace Hub via ``transformers.AutoModel``) and stacks a configurable
classifier head on top of the per-patch token sequence. The standard
V-JEPA evaluation protocol is to **freeze** the encoder and only train the
probe -- a small head with on the order of one million parameters -- which
is the default here.

**Phase 4.4 extensions (PEFT + multi-query probe):** optional
`PEFT <https://github.com/huggingface/peft>`_ LoRA adapters on attention
linear layers let a **small** set of backbone weights adapt to domain shift
while keeping the bulk of the ViT frozen. The attentive head can use
**multiple** learnable query tokens whose cross-attention outputs are
mean-pooled before the classifier, improving capacity for multi-stage verbs
without widening the backbone.

Three head types are supported:
    * ``attentive`` -- multi-query (configurable) multi-head attention pool,
      followed by ``LayerNorm`` + ``Linear``. With ``head_num_queries=1`` this
      matches the V-JEPA paper's single-query recipe.
    * ``linear``    -- ``LayerNorm`` + ``Linear`` over the mean-pooled tokens.
    * ``mean_linear`` -- ``LayerNorm`` + ``Dropout`` + ``Linear`` over the
      mean-pooled tokens (alias of ``linear`` with an extra dropout slot).

Available HF backbones (June 2025, see the V-JEPA 2 HuggingFace collection):

    ============================================  =====  ===========
    Repository                                    Params Resolution
    ============================================  =====  ===========
    facebook/vjepa2-vitl-fpc64-256                0.3 B  256
    facebook/vjepa2-vith-fpc64-256                0.7 B  256
    facebook/vjepa2-vitg-fpc64-256                1.0 B  256
    facebook/vjepa2-vitg-fpc64-384                1.0 B  384
    facebook/vjepa2-vitl-fpc16-256-ssv2           0.4 B  256  (SSv2 ft)
    facebook/vjepa2-vitg-fpc64-384-ssv2           1.0 B  384  (SSv2 ft)
    ============================================  =====  ===========

The SSv2-finetuned encoders are particularly attractive for our task: the
features have already been adapted to the Something-Something distribution,
so training a 33-class probe on top converges quickly.

Forward contract:
    Input:  ``video_batch`` -- ``(B, T, C, H, W)`` float tensor,
            ImageNet-normalized (mean ``(0.485, 0.456, 0.406)``,
            std ``(0.229, 0.224, 0.225)``). ``H = W = image_size`` of the
            chosen backbone (256 or 384). ``T`` must be even (V-JEPA's
            tubelet size is 2).
    Output: ``(B, num_classes)`` logits.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import torch
import torch.nn as nn
from omegaconf import DictConfig

from smth2smth.shared.models.registry import register_model

HeadType = Literal["attentive", "linear", "mean_linear"]

DEFAULT_HF_REPO: str = "facebook/vjepa2-vitl-fpc64-256"

_DEFAULT_LORA_TARGETS: tuple[str, ...] = ("query", "key", "value", "proj")


def _try_apply_peft_lora(
    backbone: nn.Module,
    *,
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: Sequence[str] | None,
) -> nn.Module:
    """Wrap ``backbone`` with HuggingFace PEFT LoRA adapters if ``peft`` is installed.

    Args:
        backbone: A ``transformers`` pre-trained encoder (e.g. ``VJEPA2Model``).
        r: LoRA rank.
        lora_alpha: LoRA scaling (typically equal to ``r`` or ``2*r``).
        lora_dropout: Dropout on LoRA paths.
        target_modules: Submodule name suffixes to attach adapters to (matched
            by PEFT against the module graph). Defaults to ViT-style attention
            projections.

    Returns:
        The PEFT-wrapped model (trainable adapters, frozen base weights).

    Raises:
        ImportError: If the ``peft`` package is not installed.
        RuntimeError: If PEFT cannot match any ``target_modules`` (mis-config).
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "LoRA is enabled (model.lora_enabled=true) but the `peft` package "
            "is not installed. Add it with: uv add peft  (or pip install peft)."
        ) from exc

    modules = list(target_modules) if target_modules else list(_DEFAULT_LORA_TARGETS)
    lora_config = LoraConfig(
        r=int(r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        bias="none",
        target_modules=modules,
    )
    try:
        return get_peft_model(backbone, lora_config)
    except Exception as exc:
        raise RuntimeError(
            "PEFT LoRA could not be applied to this backbone. "
            "Try adjusting model.lora_target_modules to match the encoder's "
            "attention linear names (default for HF V-JEPA~2: "
            "``query``, ``key``, ``value``, ``proj``). "
            f"Original error: {exc}"
        ) from exc


class AttentiveProbe(nn.Module):
    """Attentive pooling head with one or more learned query tokens.

    With ``num_queries=1`` this matches the V-JEPA paper's attentive probe:
    a single query cross-attends over patch tokens. With ``num_queries>1``,
    each query attends independently; outputs are **mean-pooled** across
    queries before ``LayerNorm`` and the classifier, increasing probe capacity
    for compositional actions without changing logit dimensionality.

    Args:
        feature_dim: Hidden size of the encoder's output tokens.
        num_classes: Number of output classes.
        num_heads: Multi-head attention heads. Must divide ``feature_dim``.
        num_queries: Number of learnable query tokens (>= 1).
        dropout: Dropout in the attention layer and on the pooled features.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        num_heads: int = 8,
        num_queries: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError(
                f"num_heads={num_heads} does not divide feature_dim={feature_dim}; "
                "pick a num_heads that divides the backbone's hidden_size."
            )
        nq = int(num_queries)
        if nq < 1:
            raise ValueError(f"num_queries must be >= 1, got {num_queries}")
        self.num_queries = nq
        self.query = nn.Parameter(torch.empty(1, nq, feature_dim))
        # Learnable queries: N(0, 0.01^2) (small-Gaussian probe init).
        nn.init.normal_(self.query, mean=0.0, std=0.01)
        self.attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pool a token sequence into per-class logits.

        Args:
            tokens: ``(B, N, D)`` patch tokens from the encoder.

        Returns:
            ``(B, num_classes)`` classification logits.
        """
        batch_size = tokens.size(0)
        query = self.query.expand(batch_size, -1, -1)
        pooled, _ = self.attn(query, tokens, tokens, need_weights=False)
        # (B, Q, D) -> mean over queries -> (B, D)
        pooled = pooled.mean(dim=1)
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class MeanLinearHead(nn.Module):
    """Mean-pool + LayerNorm + Dropout + Linear classifier head.

    A lightweight alternative to :class:`AttentiveProbe` that reduces the
    encoder's token sequence with a uniform mean before projecting to logits.
    Faster, has fewer parameters, and is competitive for many downstream tasks.

    Args:
        feature_dim: Hidden size of the encoder's output tokens.
        num_classes: Number of output classes.
        dropout: Dropout applied to the mean-pooled features.
    """

    def __init__(self, feature_dim: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Mean-pool the token sequence and project to logits.

        Args:
            tokens: ``(B, N, D)`` patch tokens from the encoder.

        Returns:
            ``(B, num_classes)`` classification logits.
        """
        pooled = tokens.mean(dim=1)
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def _build_head(
    head_type: HeadType,
    feature_dim: int,
    num_classes: int,
    num_heads: int,
    head_num_queries: int,
    dropout: float,
) -> nn.Module:
    """Construct the requested classifier head."""
    if head_type == "attentive":
        return AttentiveProbe(
            feature_dim=feature_dim,
            num_classes=num_classes,
            num_heads=num_heads,
            num_queries=int(head_num_queries),
            dropout=dropout,
        )
    if head_type in ("linear", "mean_linear"):
        return MeanLinearHead(
            feature_dim=feature_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
    raise ValueError(
        f"Unknown head_type: {head_type!r}. Expected one of 'attentive', 'linear', 'mean_linear'."
    )


class VJEPA2Probe(nn.Module):
    """Frozen (or PEFT-adapted) V-JEPA 2 encoder + trainable classifier probe.

    The HuggingFace ``transformers`` package is imported lazily inside the
    constructor so unit tests can mock it via ``sys.modules['transformers']``
    without requiring the heavy dependency for non-V-JEPA-2 use cases.

    Args:
        num_classes: Number of output classes.
        hf_repo: HuggingFace Hub repository id for the V-JEPA 2 backbone.
        head_type: ``"attentive"``, ``"linear"``, or ``"mean_linear"``.
        head_num_heads: Number of attention heads in the attentive head;
            must divide the backbone's ``hidden_size``. Ignored when
            ``head_type`` is not ``"attentive"``.
        head_num_queries: Number of learnable queries in the attentive head
            (ignored for linear heads). Default ``1`` recovers the paper probe.
        head_dropout: Dropout in the head.
        freeze_backbone: When ``True`` (default), pretrained encoder weights are
            frozen. If ``lora_enabled`` is also ``True``, only the **base**
            weights stay frozen; LoRA adapter weights remain trainable and the
            encoder forward runs **with** gradients through adapters.
        lora_enabled: If ``True``, wrap the backbone with PEFT LoRA (requires
            ``peft``). Recommended together with ``freeze_backbone=true``.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling.
        lora_dropout: LoRA dropout.
        lora_target_modules: List of attention submodule names to adapt; ``None``
            uses ``("q_proj", "k_proj", "v_proj", "out_proj")``.
        attn_implementation: ``transformers`` attention kernel selector.

    Attributes:
        backbone: The HuggingFace encoder (optionally a ``PeftModel``).
        head: The trainable classifier head.
        feature_dim: Hidden size of the encoder.
        hf_repo: The Hub id the backbone was loaded from.
        freeze_backbone: Whether the **pretrained** backbone weights are frozen.
        lora_enabled: Whether LoRA adapters are attached.
    """

    def __init__(
        self,
        num_classes: int,
        hf_repo: str = DEFAULT_HF_REPO,
        head_type: HeadType = "attentive",
        head_num_heads: int = 8,
        head_num_queries: int = 1,
        head_dropout: float = 0.0,
        freeze_backbone: bool = True,
        lora_enabled: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] | None = None,
        attn_implementation: str = "sdpa",
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "The 'vjepa2' model requires the 'transformers' package. "
                "Install it via `uv add transformers` (or `pip install "
                "transformers>=4.45`)."
            ) from exc

        self.hf_repo: str = str(hf_repo)
        self.freeze_backbone: bool = bool(freeze_backbone)
        self.lora_enabled: bool = bool(lora_enabled)

        if self.lora_enabled and not self.freeze_backbone:
            raise ValueError(
                "lora_enabled=true is only supported with freeze_backbone=true "
                "(PEFT keeps the pretrained core frozen while adapters train). "
                "For full fine-tuning, set lora_enabled=false."
            )

        load_kwargs: dict[str, Any] = {}
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation

        backbone = AutoModel.from_pretrained(self.hf_repo, **load_kwargs)

        if self.lora_enabled:
            tm = (
                list(lora_target_modules)
                if lora_target_modules is not None
                else None
            )
            self.backbone = _try_apply_peft_lora(
                backbone,
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                target_modules=tm,
            )
        else:
            self.backbone = backbone
            if self.freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False

        base_for_config: nn.Module = self.backbone
        if self.lora_enabled and hasattr(self.backbone, "get_base_model"):
            base_for_config = self.backbone.get_base_model()  # type: ignore[no-untyped-call]
        backbone_config = getattr(base_for_config, "config", None)
        if backbone_config is None or not hasattr(backbone_config, "hidden_size"):
            raise RuntimeError(
                f"Backbone {self.hf_repo!r} does not expose ``config.hidden_size``; "
                "this wrapper expects a HuggingFace ``VJEPA2Model``-shaped model."
            )
        self.feature_dim: int = int(backbone_config.hidden_size)

        self.head = _build_head(
            head_type=head_type,
            feature_dim=self.feature_dim,
            num_classes=int(num_classes),
            num_heads=int(head_num_heads),
            head_num_queries=int(head_num_queries),
            dropout=float(head_dropout),
        )

        # Gradients through encoder when LoRA adapters train or full FT.
        self._encoder_needs_grad: bool = bool(self.lora_enabled or not self.freeze_backbone)

    def train(self, mode: bool = True) -> VJEPA2Probe:
        """Set training mode.

        When the backbone is fully frozen (no LoRA), keep it in ``eval()`` so
        dropout / stochastic depth inside the encoder stay off. With LoRA, the
        wrapped backbone follows ``mode`` so adapter dropout behaves correctly.
        """
        super().train(mode)
        if self.freeze_backbone and not self.lora_enabled:
            self.backbone.eval()
        return self

    def _encode(self, video_batch: torch.Tensor) -> torch.Tensor:
        """Run the V-JEPA 2 encoder and return the patch-token sequence."""
        outputs = self.backbone(
            pixel_values_videos=video_batch,
            skip_predictor=True,
        )
        return outputs.last_hidden_state

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """Encode a clip with V-JEPA 2 and run the classifier probe on top."""
        if self._encoder_needs_grad:
            tokens = self._encode(video_batch)
        else:
            with torch.no_grad():
                tokens = self._encode(video_batch)
        return self.head(tokens)


@register_model("vjepa2")
def build_vjepa2(cfg: DictConfig) -> nn.Module:
    """Builder hook used by :func:`smth2smth.shared.models.registry.build_model`."""
    model_cfg = cfg.model
    lora_tm_raw = model_cfg.get("lora_target_modules")
    lora_tm: Sequence[str] | None
    if lora_tm_raw is None:
        lora_tm = None
    elif isinstance(lora_tm_raw, (list, tuple)):
        lora_tm = [str(x) for x in lora_tm_raw]
    else:
        from omegaconf import ListConfig, OmegaConf

        if isinstance(lora_tm_raw, ListConfig):
            lora_tm = [str(x) for x in OmegaConf.to_container(lora_tm_raw, resolve=True)]  # type: ignore[arg-type]
        else:
            lora_tm = None
    return VJEPA2Probe(
        num_classes=int(model_cfg.num_classes),
        hf_repo=str(model_cfg.get("hf_repo", DEFAULT_HF_REPO)),
        head_type=str(model_cfg.get("head_type", "attentive")),
        head_num_heads=int(model_cfg.get("head_num_heads", 8)),
        head_num_queries=int(model_cfg.get("head_num_queries", 1)),
        head_dropout=float(model_cfg.get("head_dropout", 0.0)),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
        lora_enabled=bool(model_cfg.get("lora_enabled", False)),
        lora_r=int(model_cfg.get("lora_r", 8)),
        lora_alpha=int(model_cfg.get("lora_alpha", 16)),
        lora_dropout=float(model_cfg.get("lora_dropout", 0.05)),
        lora_target_modules=lora_tm if isinstance(lora_tm, (list, tuple)) else None,
        attn_implementation=str(model_cfg.get("attn_implementation", "sdpa")),
    )
