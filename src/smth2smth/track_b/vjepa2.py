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
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from omegaconf import DictConfig

from smth2smth.shared.models.registry import register_model

HeadType = Literal["attentive", "linear", "mean_linear", "multi_block_attentive"]

DEFAULT_HF_REPO: str = "facebook/vjepa2-vitl-fpc64-256"
DEFAULT_SSV2FT_HF_REPO: str = "facebook/vjepa2-vitl-fpc16-256-ssv2"

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


class MultiBlockAttentiveProbe(nn.Module):
    """Meta's V-JEPA 2 attentive classifier: D self-attention blocks + cross-attention pool.

    Replicates the ``AttentiveClassifier(embed_dim=1024, num_heads=16, depth=4,
    num_classes=N)`` architecture from facebookresearch/vjepa2 (eval config
    ``num_probe_blocks: 4``, ``num_heads: 16``, used to score 73.7 % on SSv2-174
    ViT-L). Compared to :class:`AttentiveProbe` (single MHA layer), the
    multi-block stack lets the probe combine fine-grained motion cues across
    tokens before pooling, which is the structurally largest probe-side change
    available for a frozen encoder.

    Args:
        feature_dim: Hidden size of the input tokens (``K × backbone_hidden``
            when last-K-block concat is enabled upstream).
        num_classes: Output dimensionality.
        num_heads: Attention heads for both self-attention and cross-attention.
            Must divide ``feature_dim``.
        depth: Number of self-attention blocks before pooling.
        mlp_ratio: FFN expansion factor inside each self-attention block.
        dropout: Dropout in attention and FFN.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        num_heads: int = 16,
        depth: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError(
                f"num_heads={num_heads} does not divide feature_dim={feature_dim}."
            )
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}.")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=int(feature_dim * float(mlp_ratio)),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.self_attn_blocks = nn.TransformerEncoder(encoder_layer, num_layers=int(depth))
        self.query = nn.Parameter(torch.empty(1, 1, feature_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.self_attn_blocks(tokens)
        b = h.size(0)
        q = self.query.expand(b, -1, -1)
        pooled, _ = self.cross_attn(q, h, h, need_weights=False)
        pooled = pooled.squeeze(1)
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
    head_depth: int = 4,
    head_mlp_ratio: float = 4.0,
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
    if head_type == "multi_block_attentive":
        return MultiBlockAttentiveProbe(
            feature_dim=feature_dim,
            num_classes=num_classes,
            num_heads=num_heads,
            depth=int(head_depth),
            mlp_ratio=float(head_mlp_ratio),
            dropout=dropout,
        )
    if head_type in ("linear", "mean_linear"):
        return MeanLinearHead(
            feature_dim=feature_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
    raise ValueError(
        f"Unknown head_type: {head_type!r}. Expected one of 'attentive', "
        "'multi_block_attentive', 'linear', 'mean_linear'."
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
        head_depth: int = 4,
        head_mlp_ratio: float = 4.0,
        head_last_k_blocks: int = 1,
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
        self.head_last_k_blocks: int = max(1, int(head_last_k_blocks))

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
        backbone_hidden = int(backbone_config.hidden_size)
        # Last-K-block concat multiplies the channel width fed into the head.
        self.feature_dim: int = backbone_hidden * self.head_last_k_blocks

        self.head = _build_head(
            head_type=head_type,
            feature_dim=self.feature_dim,
            num_classes=int(num_classes),
            num_heads=int(head_num_heads),
            head_num_queries=int(head_num_queries),
            dropout=float(head_dropout),
            head_depth=int(head_depth),
            head_mlp_ratio=float(head_mlp_ratio),
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
        """Run the V-JEPA 2 encoder and return the patch-token sequence.

        With ``head_last_k_blocks > 1`` the channel dim is the concatenation
        of the last ``K`` encoder hidden states, which the multi-block head
        consumes directly. Default ``K == 1`` is byte-for-byte identical to
        the original behavior.
        """
        if self.head_last_k_blocks > 1:
            outputs = self.backbone(
                pixel_values_videos=video_batch,
                skip_predictor=True,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states
            if hidden_states is None or len(hidden_states) < self.head_last_k_blocks:
                raise RuntimeError(
                    f"Backbone returned {len(hidden_states) if hidden_states else 0} hidden "
                    f"states; head_last_k_blocks={self.head_last_k_blocks} requires more."
                )
            return torch.cat(list(hidden_states[-self.head_last_k_blocks :]), dim=-1)
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
    lora_tm = _resolve_lora_target_modules(model_cfg.get("lora_target_modules"))
    return VJEPA2Probe(
        num_classes=int(model_cfg.num_classes),
        hf_repo=str(model_cfg.get("hf_repo", DEFAULT_HF_REPO)),
        head_type=str(model_cfg.get("head_type", "attentive")),
        head_num_heads=int(model_cfg.get("head_num_heads", 8)),
        head_num_queries=int(model_cfg.get("head_num_queries", 1)),
        head_dropout=float(model_cfg.get("head_dropout", 0.0)),
        head_depth=int(model_cfg.get("head_depth", 4)),
        head_mlp_ratio=float(model_cfg.get("head_mlp_ratio", 4.0)),
        head_last_k_blocks=int(model_cfg.get("head_last_k_blocks", 1)),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
        lora_enabled=bool(model_cfg.get("lora_enabled", False)),
        lora_r=int(model_cfg.get("lora_r", 8)),
        lora_alpha=int(model_cfg.get("lora_alpha", 16)),
        lora_dropout=float(model_cfg.get("lora_dropout", 0.05)),
        lora_target_modules=lora_tm,
        attn_implementation=str(model_cfg.get("attn_implementation", "sdpa")),
    )


def _resolve_lora_target_modules(raw: Any) -> Sequence[str] | None:
    """Coerce a Hydra/OmegaConf value for ``lora_target_modules`` to a plain list.

    ``None``/missing returns ``None`` (callers substitute their default). Lists,
    tuples, and ``ListConfig`` are converted to ``list[str]``. Anything else
    (scalar, dict, etc.) is rejected as a config error.
    """
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    from omegaconf import ListConfig, OmegaConf

    if isinstance(raw, ListConfig):
        return [str(x) for x in OmegaConf.to_container(raw, resolve=True)]  # type: ignore[arg-type]
    return None


def _load_local_to_ssv2_idx(
    label_source_dir: Path,
    id2label: dict[int, str],
    num_classes: int,
    log_fn: Any = print,
) -> torch.Tensor:
    """Build the ``(num_classes,)`` index tensor mapping local class idx → SSv2 native idx.

    Uses the same normalized-name + token-aligned-unique-prefix matching as
    :func:`smth2smth.track_b.zero_shot.build_label_mapping`. Every local class
    folder under ``label_source_dir`` must resolve to a unique SSv2 idx; we
    raise loudly on any unmatched name so head-slice doesn't silently align to
    the wrong class. The output is indexed by *local* class index (so the i-th
    row of the sliced classifier corresponds to local class ``i``); class
    indices not present in the folder layout (e.g. local idx 027 in the 32-
    class subset) are filled with ``-1`` and must be masked downstream.
    """
    from smth2smth.track_b.zero_shot import build_label_mapping

    if not label_source_dir.is_dir():
        raise FileNotFoundError(
            f"label_source_dir={label_source_dir!s} does not exist; required to "
            "derive the local→SSv2 class-index map for head-slice."
        )
    class_dirs = sorted(p for p in label_source_dir.iterdir() if p.is_dir())
    mapping, unmatched = build_label_mapping(class_dirs, id2label, log_fn=log_fn)
    if unmatched:
        raise RuntimeError(
            "Cannot align the following local class folders to SSv2 id2label: "
            f"{unmatched!r}. Fix the folder names or extend the matcher."
        )
    if not mapping:
        raise RuntimeError(
            f"Empty local→SSv2 mapping derived from {label_source_dir!s}."
        )
    idx_tensor = torch.full((num_classes,), -1, dtype=torch.long)
    for local_idx, ssv2_idx in mapping.items():
        if 0 <= local_idx < num_classes:
            idx_tensor[local_idx] = int(ssv2_idx)
    n_resolved = int((idx_tensor >= 0).sum().item())
    log_fn(
        f"[ssv2ft] head-slice: resolved {n_resolved}/{num_classes} local class indices "
        f"({len(mapping)} folder matches)"
    )
    return idx_tensor


class VJEPA2SSv2FTProbe(nn.Module):
    """V-JEPA 2 SSv2-finetuned classifier with head-sliced, optionally LoRA-adapted encoder.

    Loads :class:`transformers.VJEPA2ForVideoClassification` from an SSv2-
    finetuned checkpoint (e.g. ``facebook/vjepa2-vitl-fpc16-256-ssv2``), keeps
    Meta's pretrained attentive pooler, and replaces the 174-class linear
    classifier with one whose rows are *initialized from the corresponding 32
    rows of Meta's classifier*. This imports the supervised SSv2 class
    prototypes for our subset, so the model starts well above random and far
    above an SSL-only baseline. The encoder is wrapped with PEFT LoRA when
    ``lora_enabled`` is set; the pooler and classifier are always trainable.

    Forward: ``(B, T, C, H, W)`` → ``(B, num_classes)`` logits via
    ``VJEPA2ForVideoClassification.forward(pixel_values_videos=...).logits``.

    Args:
        num_classes: Number of local class indices in our subset.
        hf_repo: SSv2-finetuned HuggingFace repo.
        label_source_dir: Path whose subdirectories are ``NNN_<SSv2-label>``
            class folders; used at construction time to derive the
            local→SSv2 index map for head-slicing. Required when
            ``init_head_from_ssv2`` is true.
        init_head_from_ssv2: When ``True`` (default), slice Meta's 174-class
            classifier rows into a fresh ``nn.Linear(hidden, num_classes)``.
            Set ``False`` for inference-time reconstruction from a saved
            checkpoint (random head; the saved state dict overlays it).
        freeze_backbone_base: Keep the pretrained encoder weights frozen
            (LoRA adapter weights remain trainable when ``lora_enabled``).
        lora_enabled, lora_r, lora_alpha, lora_dropout, lora_target_modules:
            Standard PEFT knobs (see :func:`_try_apply_peft_lora`).
        attn_implementation: ``transformers`` attention backend.
    """

    def __init__(
        self,
        num_classes: int,
        hf_repo: str = DEFAULT_SSV2FT_HF_REPO,
        *,
        label_source_dir: str | Path | None = None,
        init_head_from_ssv2: bool = True,
        freeze_backbone_base: bool = True,
        lora_enabled: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] | None = None,
        attn_implementation: str = "sdpa",
    ) -> None:
        super().__init__()
        try:
            from transformers import VJEPA2ForVideoClassification
        except ImportError as exc:
            raise ImportError(
                "The 'vjepa2_ssv2ft' model requires the 'transformers' package."
            ) from exc

        self.hf_repo: str = str(hf_repo)
        self.lora_enabled: bool = bool(lora_enabled)
        self.freeze_backbone_base: bool = bool(freeze_backbone_base)

        load_kwargs: dict[str, Any] = {}
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        model = VJEPA2ForVideoClassification.from_pretrained(self.hf_repo, **load_kwargs)

        hidden = int(model.config.hidden_size)
        self.feature_dim: int = hidden

        if init_head_from_ssv2:
            if label_source_dir is None:
                raise ValueError(
                    "init_head_from_ssv2=True requires label_source_dir to derive "
                    "the local→SSv2 class index map."
                )
            id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
            idx_tensor = _load_local_to_ssv2_idx(
                Path(str(label_source_dir)).resolve(),
                id2label,
                int(num_classes),
            )
            self.register_buffer("local_to_ssv2_idx", idx_tensor, persistent=True)
            new_head = nn.Linear(hidden, int(num_classes), bias=True)
            with torch.no_grad():
                resolved = idx_tensor >= 0
                if resolved.any():
                    sel = idx_tensor.clone()
                    sel[~resolved] = 0  # safe gather; unresolved rows overwritten below
                    new_head.weight.data.copy_(model.classifier.weight.data.index_select(0, sel))
                    new_head.bias.data.copy_(model.classifier.bias.data.index_select(0, sel))
                if (~resolved).any():
                    # Local indices without a SSv2 match (e.g. missing class 027): random init.
                    nn.init.normal_(new_head.weight.data[~resolved], mean=0.0, std=0.01)
                    nn.init.zeros_(new_head.bias.data[~resolved])
        else:
            self.register_buffer(
                "local_to_ssv2_idx",
                torch.full((int(num_classes),), -1, dtype=torch.long),
                persistent=True,
            )
            new_head = nn.Linear(hidden, int(num_classes), bias=True)
            nn.init.normal_(new_head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(new_head.bias)

        model.classifier = new_head
        model.num_labels = int(num_classes)
        model.config.num_labels = int(num_classes)

        if self.freeze_backbone_base:
            for param in model.vjepa2.parameters():
                param.requires_grad = False

        if self.lora_enabled:
            tm = list(lora_target_modules) if lora_target_modules is not None else list(_DEFAULT_LORA_TARGETS)
            model.vjepa2 = _try_apply_peft_lora(
                model.vjepa2,
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                target_modules=tm,
            )

        self.model = model

    def train(self, mode: bool = True) -> VJEPA2SSv2FTProbe:
        """Set training mode but keep a fully-frozen encoder in eval (no LoRA)."""
        super().train(mode)
        if self.freeze_backbone_base and not self.lora_enabled:
            # Encoder stays in eval to avoid stochastic depth / dropout drift.
            base = self.model.vjepa2
            if hasattr(base, "get_base_model"):
                base.get_base_model().eval()  # type: ignore[no-untyped-call]
            else:
                base.eval()
        return self

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """Run the SSv2-finetuned classifier on a clip batch."""
        outputs = self.model(pixel_values_videos=video_batch)
        return outputs.logits


@register_model("vjepa2_ssv2ft")
def build_vjepa2_ssv2ft(cfg: DictConfig) -> nn.Module:
    """Builder for :class:`VJEPA2SSv2FTProbe` (SSv2-FT checkpoint + head-slice + LoRA)."""
    model_cfg = cfg.model
    lora_tm = _resolve_lora_target_modules(model_cfg.get("lora_target_modules"))

    # ``label_source_dir`` defaults to the dataset's train_dir so the model can
    # auto-derive the local→SSv2 mapping at construction time. Override at the
    # CLI or in the config to point at any folder whose class names match.
    default_label_source = None
    if hasattr(cfg, "dataset"):
        default_label_source = cfg.dataset.get("train_dir", None)
    label_source_dir = model_cfg.get("label_source_dir", default_label_source)

    return VJEPA2SSv2FTProbe(
        num_classes=int(model_cfg.num_classes),
        hf_repo=str(model_cfg.get("hf_repo", DEFAULT_SSV2FT_HF_REPO)),
        label_source_dir=label_source_dir,
        init_head_from_ssv2=bool(model_cfg.get("init_head_from_ssv2", True)),
        freeze_backbone_base=bool(model_cfg.get("freeze_backbone", True)),
        lora_enabled=bool(model_cfg.get("lora_enabled", False)),
        lora_r=int(model_cfg.get("lora_r", 16)),
        lora_alpha=int(model_cfg.get("lora_alpha", 32)),
        lora_dropout=float(model_cfg.get("lora_dropout", 0.05)),
        lora_target_modules=lora_tm,
        attn_implementation=str(model_cfg.get("attn_implementation", "sdpa")),
    )
