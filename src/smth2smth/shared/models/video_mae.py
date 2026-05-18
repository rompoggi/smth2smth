"""VideoMAE-style Video Vision Transformer (Track A, from scratch).

Follows VideoMAE (Tong et al., NeurIPS 2022) adapted for T=4 frames:
  - 3D cube embedding: Conv3d(3, D, kernel=(2,16,16)) → 392 tokens per clip
  - Learnable 3D positional encoding (n_t × n_h × n_w positions)
  - ViT-B: 12 pre-norm blocks (D=768, 12 heads, MLP 3072) — 86 M params
  - ViT-L: 24 pre-norm blocks (D=1024, 16 heads, MLP 4096) — 307 M params
  - Classification head: mean-pool all tokens → dropout → Linear  (head="mean")
                       or attentive probe (1 learnable query, cross-attn) → Linear (head="attn")

SSL pretraining (VideoMAE):
  - Tube masking at 75% ratio: same spatial positions masked across all frames
    (90% is the Kinetics setting; 75% is used here because T=4 gives fewer temporal
    tokens and 90% would leave only ~40 visible tokens out of 392)
  - Lightweight decoder: 4 pre-norm blocks (D=384, 6 heads, MLP 1536)
  - Reconstruction target: per-cube normalized pixel values
  - MSE loss only at masked positions

No flip augmentation should be used with SSv2 (direction-sensitive).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.checkpoint import checkpoint

from smth2smth.shared.models.registry import register_model

# ── Positional embedding resize (resolution / temporal transfer) ───────────────


def interpolate_pos_embed(
    pos_embed: torch.Tensor,
    *,
    src_num_frames: int,
    src_img_size: int,
    dst_num_frames: int,
    dst_img_size: int,
    tube_t: int = 2,
    patch_size: int = 16,
) -> torch.Tensor:
    """Trilinearly resize a VideoMAE ``pos_embed`` to a new spatiotemporal grid.

    Args:
        pos_embed: ``(1, N_src, D)`` learnable positional encoding.
        src_num_frames / src_img_size: grid the checkpoint was trained with.
        dst_num_frames / dst_img_size: target supervised or pretrain grid.

    Returns:
        ``(1, N_dst, D)`` tensor on the same device/dtype as ``pos_embed``.
    """
    if pos_embed.ndim != 3 or pos_embed.shape[0] != 1:
        raise ValueError(f"pos_embed must be (1, N, D), got {tuple(pos_embed.shape)}")
    dim = pos_embed.shape[-1]
    n_t_old = src_num_frames // tube_t
    n_h_old = src_img_size // patch_size
    n_w_old = src_img_size // patch_size
    n_t_new = dst_num_frames // tube_t
    n_h_new = dst_img_size // patch_size
    n_w_new = dst_img_size // patch_size
    expected_old = n_t_old * n_h_old * n_w_old
    if pos_embed.shape[1] != expected_old:
        raise ValueError(
            f"pos_embed length {pos_embed.shape[1]} != "
            f"{expected_old} for src T={src_num_frames} size={src_img_size}."
        )
    pe = pos_embed.reshape(1, n_t_old, n_h_old, n_w_old, dim).permute(0, 4, 1, 2, 3)
    pe = F.interpolate(
        pe,
        size=(n_t_new, n_h_new, n_w_new),
        mode="trilinear",
        align_corners=False,
    )
    return pe.permute(0, 2, 3, 4, 1).reshape(1, n_t_new * n_h_new * n_w_new, dim)


# ── Building blocks ────────────────────────────────────────────────────────────


class DropPath(nn.Module):
    """Per-sample stochastic depth. Identity at eval time."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}")
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob) / keep_prob
        return x * mask


class PatchEmbed3D(nn.Module):
    """3D cube embedding via Conv3d(kernel=(tube_t, patch_size, patch_size))."""

    def __init__(
        self,
        num_frames: int = 4,
        img_size: int = 224,
        tube_t: int = 2,
        patch_size: int = 16,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        if num_frames % tube_t != 0:
            raise ValueError(f"num_frames={num_frames} must be divisible by tube_t={tube_t}")
        if img_size % patch_size != 0:
            raise ValueError(f"img_size={img_size} must be divisible by patch_size={patch_size}")
        self.n_t = num_frames // tube_t
        self.n_h = img_size // patch_size
        self.n_w = img_size // patch_size
        self.num_tokens = self.n_t * self.n_h * self.n_w
        self.proj = nn.Conv3d(
            3, embed_dim,
            kernel_size=(tube_t, patch_size, patch_size),
            stride=(tube_t, patch_size, patch_size),
        )
        w = self.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) → (B, num_tokens, embed_dim)
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)        # (B, C, T, H, W)
        x = self.proj(x)                     # (B, D, n_t, n_h, n_w)
        return x.flatten(2).transpose(1, 2)  # (B, n_t*n_h*n_w, D)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    """Pre-norm ViT block: LN → MHA → residual, LN → MLP → residual."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ── Hyper-Connections (HC) and Manifold-Constrained HC (mHC) ───────────────────
#
# Implements Zhu et al. *Hyper-Connections* (ICLR 2025, arXiv:2409.19606) and
# Xie et al. *Manifold-Constrained Hyper-Connections* (2025, arXiv:2512.24880),
# Static variant only (SHC / mHC), no input-dependent dynamic routing.
#
# Per-sublayer F (Attn or MLP), HC widens the residual stream into n parallel
# "hyper-hidden" vectors stacked as H ∈ R^{B × n × T × D} and updates them as:
#     h_pre = einsum("n,bntd->btd", alpha_pre, H)
#     y     = F(LayerNorm(h_pre))
#     H_new = einsum("ij,bjtd->bitd", M, H)
#           + einsum("n,btd->bntd", beta, y)
# Identity init (M=I, alpha_pre=beta=e_0) makes step 0 exactly pre-norm.
#
# mHC replaces the unconstrained M with the Sinkhorn-Knopp projection of a
# raw learnable matrix M_raw onto the Birkhoff polytope (doubly stochastic
# matrices); SK exp/normalize is done in fp32 even under AMP bf16 (NaN risk).


def sinkhorn_knopp(
    m_raw: torch.Tensor, *, k_iters: int = 3, tau: float = 1.0, eps: float = 1e-8
) -> torch.Tensor:
    """Project ``m_raw`` to the Birkhoff polytope via Sinkhorn-Knopp.

    The exp/normalize is computed in fp32 for numerical stability under AMP bf16
    (under bf16 the exponential easily saturates; the doc lists this as a hard
    rule). Caller is responsible for casting the result back to the activation
    dtype.

    Args:
        m_raw: ``(n, n)`` raw learnable mixing matrix.
        k_iters: Number of SK iterations. The paper uses K=20 at LLM scale;
            K=3 is sufficient at n=4 (converges to <1e-3 deviation).
        tau: Temperature scaling before exp. ``tau=1.0`` matches the doc.
        eps: Numerical floor for row/column sums.

    Returns:
        ``(n, n)`` doubly-stochastic tensor in fp32.
    """
    if k_iters < 0:
        raise ValueError(f"k_iters must be >= 0, got {k_iters}.")
    m = (m_raw.float() / float(tau)).exp()
    for _ in range(k_iters):
        m = m / (m.sum(dim=1, keepdim=True) + eps)
        m = m / (m.sum(dim=0, keepdim=True) + eps)
    return m


class HCRouter(nn.Module):
    """Static Hyper-Connections router for a single sublayer.

    Holds the per-sublayer mixing matrix ``M`` (or its raw form ``M_raw`` in
    mHC mode, projected through Sinkhorn-Knopp on every forward), the depth-mix
    read ``alpha_pre``, and the write weights ``beta``.

    Forward signature: given ``H ∈ (B, n, T, D)`` and a callable ``sublayer``
    that maps ``(B, T, D) → (B, T, D)`` (with its own LayerNorm applied to the
    pre-mixed vector), returns the updated ``H``.

    Identity-equivalent initialization (HC §3.4):
        - SHC: ``M = I_n``,    ``alpha_pre = beta = e_0``
        - mHC: ``M_raw = c·I_n`` with ``c`` large so ``SK(M_raw) ≈ I``,
                 ``alpha_pre = beta = e_0``.

    HC scalars (M, M_raw, alpha_pre, beta) are excluded from weight-decay and
    layer-wise LR decay by the training pipeline (treated like LayerNorm gains).
    """

    def __init__(
        self,
        n: int,
        *,
        variant: str = "static",
        sk_iters: int = 3,
        sk_tau: float = 1.0,
        diagonal_init: float = 10.0,
    ) -> None:
        super().__init__()
        if n < 1:
            raise ValueError(f"HC expansion rate n must be >= 1, got {n}.")
        if variant not in {"static", "mhc"}:
            raise ValueError(f"HCRouter variant must be 'static' or 'mhc', got {variant!r}.")
        self.n = int(n)
        self.variant = variant
        self.sk_iters = int(sk_iters)
        self.sk_tau = float(sk_tau)

        # Depth-mix read weights (n,): identity init = e_0 (one-hot at index 0).
        e0 = torch.zeros(n)
        e0[0] = 1.0
        self.alpha_pre = nn.Parameter(e0.clone())
        self.beta = nn.Parameter(e0.clone())

        if variant == "static":
            # Unconstrained mixing matrix. Identity init.
            self.M = nn.Parameter(torch.eye(n))
        else:
            # mHC: raw matrix, projected via SK on every forward.  Large diagonal
            # init makes SK(M_raw) ≈ I at step 0 (with c=10, diag ≈ 0.99982).
            self.M_raw = nn.Parameter(diagonal_init * torch.eye(n))

    @property
    def is_mhc(self) -> bool:
        return self.variant == "mhc"

    def mixing_matrix(self, dtype: torch.dtype) -> torch.Tensor:
        """Return the effective mixing matrix in ``dtype``.

        For SHC this is just ``M`` (cast). For mHC we compute SK(M_raw) in fp32
        (per the doc's hard rule about AMP bf16) and cast at the end.
        """
        if self.variant == "static":
            return self.M.to(dtype=dtype)
        m = sinkhorn_knopp(self.M_raw, k_iters=self.sk_iters, tau=self.sk_tau)
        return m.to(dtype=dtype)

    def forward(self, H: torch.Tensor, sublayer_fn) -> torch.Tensor:
        """Update the hyper-state ``H`` through one sublayer.

        Args:
            H: ``(B, n, T, D)`` hyper-hidden states.
            sublayer_fn: Callable taking ``(B, T, D)`` and returning ``(B, T, D)``.
                Typically ``lambda v: drop_path(F(LN(v)))``.
        Returns:
            ``(B, n, T, D)`` updated hyper-state.
        """
        if H.dim() != 4 or H.shape[1] != self.n:
            raise ValueError(
                f"HCRouter expected H of shape (B, {self.n}, T, D), got {tuple(H.shape)}."
            )

        # Depth-mix: read one vector from H using alpha_pre. (B, T, D).
        alpha_pre = self.alpha_pre.to(H.dtype)
        h_pre = torch.einsum("n,bntd->btd", alpha_pre, H)

        # Run the sublayer (Attn or MLP, including its own LN + DropPath).
        y = sublayer_fn(h_pre)

        # Cross-stream propagation + write-back.
        m = self.mixing_matrix(H.dtype)
        beta = self.beta.to(H.dtype)
        H_mixed = torch.einsum("ij,bjtd->bitd", m, H)
        H_write = torch.einsum("n,btd->bntd", beta, y)
        return H_mixed + H_write


class HCTransformerBlock(nn.Module):
    """Pre-norm ViT block wrapped with HC / mHC routing on both sublayers.

    Replaces the plain pre-norm residual ``x = x + Attn(LN(x))`` /
    ``x = x + MLP(LN(x))`` with the HC update rule (see :class:`HCRouter`).

    The block consumes and returns ``H ∈ (B, n, T, D)``; it is the encoder's
    job to expand the patch-embedded tokens to ``H`` on entry and collapse
    ``H`` back to a single token stream on exit.

    Both sublayers are wrapped by their own :class:`HCRouter` so ``M``,
    ``alpha_pre`` and ``beta`` are not shared across Attn and MLP — this
    matches the HC paper's "each sublayer F gets its own router" recipe.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        *,
        n: int = 4,
        variant: str = "static",
        sk_iters: int = 3,
        sk_tau: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path)
        self.attn_router = HCRouter(
            n, variant=variant, sk_iters=sk_iters, sk_tau=sk_tau,
        )
        self.mlp_router = HCRouter(
            n, variant=variant, sk_iters=sk_iters, sk_tau=sk_tau,
        )

    def _attn_sublayer(self, v: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(v)
        out, _ = self.attn(normed, normed, normed, need_weights=False)
        return self.drop_path(out)

    def _mlp_sublayer(self, v: torch.Tensor) -> torch.Tensor:
        return self.drop_path(self.mlp(self.norm2(v)))

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        H = self.attn_router(H, self._attn_sublayer)
        H = self.mlp_router(H, self._mlp_sublayer)
        return H


# ── Encoder ────────────────────────────────────────────────────────────────────


class VideoMAEEncoder(nn.Module):
    """ViT encoder with 3D cube embedding and learnable positional encoding.

    Accepts an optional ``ids_keep`` tensor to operate only on visible tokens
    (used during VideoMAE pretraining). Pass ``None`` for full-sequence mode
    (supervised fine-tuning / inference).

    Set ``residual_variant`` to ``"shc"`` (Static Hyper-Connections) or
    ``"mhc"`` (Manifold-Constrained HC) to wrap every block's two sublayers
    with HC routing. ``"prenorm"`` (default) is the unmodified Pre-Norm path.
    Under SHC/mHC the encoder maintains a hyper-state ``H ∈ (B, n, T, D)``
    through the block stack and collapses it back via a learnable
    ``alpha_out`` on exit so the rest of the model is unchanged.

    Identity-equivalent init: at step 0, HC reduces exactly to Pre-Norm
    (only stream 0 is read/written; ``M = I`` keeps other streams as the
    initial token embedding).
    """

    def __init__(
        self,
        num_frames: int = 4,
        img_size: int = 224,
        tube_t: int = 2,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        gradient_checkpointing: bool = False,
        residual_variant: str = "prenorm",
        hc_n: int = 4,
        hc_sk_iters: int = 3,
        hc_sk_tau: float = 1.0,
    ) -> None:
        super().__init__()
        if residual_variant not in {"prenorm", "shc", "mhc"}:
            raise ValueError(
                f"residual_variant must be 'prenorm', 'shc' or 'mhc', got "
                f"{residual_variant!r}."
            )
        self.embed_dim = embed_dim
        self.residual_variant = residual_variant
        self.hc_n = int(hc_n) if residual_variant != "prenorm" else 1
        self.patch_embed = PatchEmbed3D(num_frames, img_size, tube_t, patch_size, embed_dim)
        self.num_tokens = self.patch_embed.num_tokens
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))

        dpr = [drop_path_rate * i / max(1, depth - 1) for i in range(depth)]
        if residual_variant == "prenorm":
            self.blocks = nn.ModuleList([
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dpr[i])
                for i in range(depth)
            ])
            self.alpha_out: nn.Parameter | None = None
        else:
            hc_kwargs = dict(
                n=self.hc_n,
                variant="static" if residual_variant == "shc" else "mhc",
                sk_iters=hc_sk_iters,
                sk_tau=hc_sk_tau,
            )
            self.blocks = nn.ModuleList([
                HCTransformerBlock(embed_dim, num_heads, mlp_ratio, dpr[i], **hc_kwargs)
                for i in range(depth)
            ])
            # alpha_out: read weights to collapse H back to a single stream.
            # Identity init = e_0 (one-hot at index 0) ⇒ exit equals stream 0,
            # which under M=I and alpha_pre/beta=e_0 is exactly the Pre-Norm output.
            e0 = torch.zeros(self.hc_n)
            e0[0] = 1.0
            self.alpha_out = nn.Parameter(e0.clone())
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        ids_keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)
            ids_keep: (B, n_visible) indices of visible tokens, or None for all.
        Returns:
            (B, n_visible, embed_dim) — normalized encoder output.
        """
        tokens = self.patch_embed(x) + self.pos_embed  # (B, N, D)

        if ids_keep is not None:
            tokens = torch.gather(
                tokens, 1,
                ids_keep.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
            )

        if self.residual_variant == "prenorm":
            for block in self.blocks:
                if self.gradient_checkpointing and self.training:
                    tokens = checkpoint(block, tokens, use_reentrant=False)
                else:
                    tokens = block(tokens)
            return self.norm(tokens)

        # HC / mHC path: widen the residual stream to (B, n, T, D), propagate
        # through HC-wrapped blocks, collapse back via alpha_out.
        H = tokens.unsqueeze(1).expand(-1, self.hc_n, -1, -1).contiguous()
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                H = checkpoint(block, H, use_reentrant=False)
            else:
                H = block(H)
        alpha_out = self.alpha_out.to(H.dtype)
        collapsed = torch.einsum("n,bntd->btd", alpha_out, H)
        return self.norm(collapsed)


# ── Classification heads ───────────────────────────────────────────────────────


class AttentiveProbeHead(nn.Module):
    """Single learnable query cross-attending to all encoder tokens.

    Implements the attentive probe used in V-JEPA 2: one learnable query token
    attends (multi-head cross-attention) to the full encoder token sequence,
    then the output is layer-normed and fed to the linear classifier.  Unlike
    mean pooling this can learn to selectively weight spatiotemporal tokens.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}."
            )
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, N, D) → (B, D) pooled feature."""
        q = self.query.expand(tokens.size(0), -1, -1)     # (B, 1, D)
        out, _ = self.attn(q, tokens, tokens, need_weights=False)
        return self.norm(out.squeeze(1))                   # (B, D)


# ── Supervised model ───────────────────────────────────────────────────────────


class VideoMAEViT(nn.Module):
    """VideoMAE ViT + classification head for supervised training.

    Supports two pooling heads (``head`` arg):
      - ``"mean"``: global average pool over all tokens → Linear (VideoMAE paper)
      - ``"attn"``: attentive probe — 1 learnable query cross-attends to all
                    tokens → Linear (V-JEPA 2 style)

    Suitable for:
      - From-scratch supervised baseline (step 1 in Track A plan)
      - Fine-tuning after VideoMAE SSL pretraining (steps 2–3)
    """

    def __init__(
        self,
        num_classes: int,
        num_frames: int = 4,
        img_size: int = 224,
        tube_t: int = 2,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        dropout: float = 0.0,
        head: str = "mean",
        head_num_heads: int = 4,
        gradient_checkpointing: bool = False,
        residual_variant: str = "prenorm",
        hc_n: int = 4,
        hc_sk_iters: int = 3,
        hc_sk_tau: float = 1.0,
    ) -> None:
        super().__init__()
        if head not in {"mean", "attn"}:
            raise ValueError(f"head must be 'mean' or 'attn', got {head!r}.")
        self.encoder = VideoMAEEncoder(
            num_frames=num_frames,
            img_size=img_size,
            tube_t=tube_t,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            gradient_checkpointing=gradient_checkpointing,
            residual_variant=residual_variant,
            hc_n=hc_n,
            hc_sk_iters=hc_sk_iters,
            hc_sk_tau=hc_sk_tau,
        )
        self.head_kind = head
        self.attn_pool: AttentiveProbeHead | None = (
            AttentiveProbeHead(embed_dim, head_num_heads) if head == "attn" else None
        )
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        features = self.encoder(x)  # (B, N, D) — already layer-normed
        if self.attn_pool is not None:
            pooled = self.attn_pool(features)   # (B, D)
        else:
            pooled = features.mean(dim=1)       # (B, D)
        return self.classifier(self.dropout(pooled))

    def freeze_backbone_for_classifier_tune(self) -> None:
        """Freeze encoder + attentive pool; train classifier only (cRT stage 2)."""
        for name, param in self.named_parameters():
            param.requires_grad = name.startswith("classifier.")


# ── ViT variant table ──────────────────────────────────────────────────────────

_VIT_VARIANTS: dict[str, dict] = {
    "vit_s": dict(embed_dim=384,  depth=12, num_heads=6),
    "vit_b": dict(embed_dim=768,  depth=12, num_heads=12),
    "vit_l": dict(embed_dim=1024, depth=24, num_heads=16),
}


# ── SSL helpers ────────────────────────────────────────────────────────────────


def make_running_cell_mask(
    batch_size: int,
    n_t: int,
    n_h: int,
    n_w: int,
    keep_ratio: float = 0.50,
    cell_h: int = 2,
    cell_w: int = 2,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Sample VideoMAE-V2 running-cell decoder mask.

    The (n_h, n_w) spatial grid is tiled into (cell_h, cell_w) cells; each
    temporal slice is masked independently. Within every (1, cell_h, cell_w)
    cell, ``round(cell_h*cell_w*keep_ratio)`` positions are kept at random.
    Returns global token indices (sorted ascending) of decoder-kept positions.

    See VideoMAE V2 paper Fig. 2 / Sec. 3.2 (arXiv:2303.16727). Keeping a fixed
    count per cell makes the per-sample decoder length deterministic, which is
    required for batched processing.
    """
    if n_h % cell_h != 0 or n_w % cell_w != 0:
        raise ValueError(
            f"running-cell mask requires n_h ({n_h}) divisible by cell_h ({cell_h}) "
            f"and n_w ({n_w}) divisible by cell_w ({cell_w})."
        )
    nh_c = n_h // cell_h
    nw_c = n_w // cell_w
    cell_size = cell_h * cell_w
    keep_per_cell = max(1, int(round(cell_size * keep_ratio)))

    noise = torch.rand(batch_size, n_t, nh_c, nw_c, cell_size, device=device)
    intra_keep = noise.argsort(dim=-1)[..., :keep_per_cell]  # (B, n_t, nh_c, nw_c, keep_per_cell)

    ih_off = (torch.arange(nh_c, device=device) * cell_h).view(1, 1, nh_c, 1, 1)
    iw_off = (torch.arange(nw_c, device=device) * cell_w).view(1, 1, 1, nw_c, 1)
    ih_global = ih_off + intra_keep // cell_w
    iw_global = iw_off + intra_keep % cell_w
    t_idx = torch.arange(n_t, device=device).view(1, n_t, 1, 1, 1)
    ids = t_idx * (n_h * n_w) + ih_global * n_w + iw_global  # (B, n_t, nh_c, nw_c, keep_per_cell)
    ids = ids.reshape(batch_size, -1).sort(dim=1).values
    return ids


def make_tube_mask(
    batch_size: int,
    n_t: int,
    n_h: int,
    n_w: int,
    mask_ratio: float = 0.90,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate tube masks (same spatial positions masked across all frames).

    Returns:
        ids_keep: (B, n_t * n_keep_spatial) — indices of visible tokens
        ids_mask: (B, n_t * n_mask_spatial) — indices of masked tokens
    """
    n_spatial = n_h * n_w
    n_keep_spatial = max(1, int(round(n_spatial * (1.0 - mask_ratio))))
    n_mask_spatial = n_spatial - n_keep_spatial

    # Per-sample random spatial permutation → same mask replicated across time
    noise = torch.rand(batch_size, n_spatial, device=device)
    ids_spatial = noise.argsort(dim=1)                         # (B, n_spatial)
    ids_keep_spatial = ids_spatial[:, :n_keep_spatial]         # (B, n_keep_spatial)
    ids_mask_spatial = ids_spatial[:, n_keep_spatial:]         # (B, n_mask_spatial)

    # Expand across temporal dimension: token idx = t * n_spatial + spatial_idx
    t_offsets = torch.arange(n_t, device=device) * n_spatial   # (n_t,)

    # (B, n_keep_spatial, 1) + (1, 1, n_t) → (B, n_keep_spatial, n_t) → (B, n_t*n_keep_spatial)
    ids_keep = (
        ids_keep_spatial.unsqueeze(-1) + t_offsets.view(1, 1, n_t)
    ).reshape(batch_size, n_t * n_keep_spatial)

    ids_mask = (
        ids_mask_spatial.unsqueeze(-1) + t_offsets.view(1, 1, n_t)
    ).reshape(batch_size, n_t * n_mask_spatial)

    # Sort so token order matches positional embedding order
    ids_keep = ids_keep.sort(dim=1).values
    ids_mask = ids_mask.sort(dim=1).values

    return ids_keep, ids_mask


def patchify(
    clips: torch.Tensor,
    tube_t: int = 2,
    patch_size: int = 16,
) -> torch.Tensor:
    """Rearrange video clips into per-cube pixel vectors.

    Args:
        clips: (B, T, C, H, W)
    Returns:
        (B, n_t*n_h*n_w, 3*tube_t*patch_size*patch_size)
    """
    B, T, C, H, W = clips.shape
    n_t = T // tube_t
    n_h = H // patch_size
    n_w = W // patch_size
    cube_dim = C * tube_t * patch_size * patch_size  # 3*2*16*16 = 1536

    x = clips.permute(0, 2, 1, 3, 4)              # (B, C, T, H, W)
    x = x.reshape(B, C, n_t, tube_t, n_h, patch_size, n_w, patch_size)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)         # (B, n_t, n_h, n_w, C, tube_t, ph, pw)
    x = x.reshape(B, n_t * n_h * n_w, cube_dim)   # (B, N, cube_dim)
    return x


# ── SSL decoder ────────────────────────────────────────────────────────────────


class VideoMAEDecoder(nn.Module):
    """Lightweight VideoMAE decoder: 4 pre-norm blocks, dim=384.

    Takes the encoder's visible-token output, reconstructs the full sequence
    by inserting a learnable mask token, then predicts pixel values for every
    cube.  Only the predictions at masked positions are used for the loss.
    """

    DECODER_DIM = 384
    DECODER_HEADS = 6
    DECODER_DEPTH = 4

    def __init__(
        self,
        encoder_dim: int,
        num_tokens: int,
        tube_t: int = 2,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        d = self.DECODER_DIM
        self.num_tokens = num_tokens
        self.proj = nn.Linear(encoder_dim, d)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_tokens, d))
        dpr = [0.0] * self.DECODER_DEPTH
        self.blocks = nn.ModuleList([
            TransformerBlock(d, self.DECODER_HEADS, mlp_ratio=4.0, drop_path=dpr[i])
            for i in range(self.DECODER_DEPTH)
        ])
        self.norm = nn.LayerNorm(d)
        cube_dim = 3 * tube_t * patch_size * patch_size
        self.head = nn.Linear(d, cube_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        visible_tokens: torch.Tensor,
        ids_keep: torch.Tensor,
        ids_mask: torch.Tensor,
        ids_decoder_kept: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            visible_tokens:   (B, n_visible, encoder_dim)
            ids_keep:         (B, n_visible) encoder-visible token indices
            ids_mask:         (B, n_masked)  encoder-masked token indices
            ids_decoder_kept: (B, k) optional running-cell decoder kept positions
                (VideoMAE-V2 dual masking). If ``None`` the decoder processes
                all N positions (V1 behavior); predictions are returned at
                ``ids_mask``. If provided, the decoder operates on the k
                decoder-kept positions and predictions are returned at those
                positions — the caller is responsible for slicing/masking the
                non-encoder-masked subset (the loss does this).

        Returns:
            (pred, ids_predict) where ``pred`` is ``(B, P, cube_dim)`` and
            ``ids_predict`` are the global token indices for which ``pred``
            holds predictions: P=n_masked (V1) or P=k (V2).
        """
        B = visible_tokens.shape[0]
        N = self.num_tokens
        d = self.mask_token.shape[-1]

        tokens = self.proj(visible_tokens)  # (B, n_visible, d)

        # Build the full canvas: visible_tokens at ids_keep, mask_token elsewhere.
        # Under autocast, `tokens` may be fp16/bf16 while parameters stay fp32; align dtypes
        # (and device) for scatter_ / addition.
        full = (
            self.mask_token.to(device=tokens.device, dtype=tokens.dtype)
            .expand(B, N, -1)
            .clone()
        )
        idx_v = ids_keep.unsqueeze(-1).expand(-1, -1, d)
        full.scatter_(1, idx_v, tokens)

        full = full + self.decoder_pos_embed.to(dtype=full.dtype)

        if ids_decoder_kept is not None:
            # Dual masking: decoder operates only on decoder-kept positions.
            idx_d = ids_decoder_kept.unsqueeze(-1).expand(-1, -1, d)
            full = torch.gather(full, 1, idx_d)  # (B, k, d)
            ids_predict = ids_decoder_kept
        else:
            ids_predict = ids_mask

        for block in self.blocks:
            full = block(full)
        full = self.norm(full)

        if ids_decoder_kept is None:
            idx_m = ids_mask.unsqueeze(-1).expand(-1, -1, d)
            full = torch.gather(full, 1, idx_m)   # (B, n_masked, d)

        return self.head(full), ids_predict


# ── SSL pretrain model ─────────────────────────────────────────────────────────


class VideoMAEPretrainModel(nn.Module):
    """Encoder + decoder for VideoMAE masked-reconstruction pretraining.

    Supports VideoMAE-V1 (encoder-only tube masking) and VideoMAE-V2 dual
    masking (encoder tube mask + decoder running-cell mask). Toggle via
    ``dual_masking=True`` and ``decoder_keep_ratio`` (V2 default 0.50).
    """

    def __init__(
        self,
        num_frames: int = 4,
        img_size: int = 224,
        tube_t: int = 2,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        mask_ratio: float = 0.75,
        gradient_checkpointing: bool = False,
        dual_masking: bool = False,
        decoder_keep_ratio: float = 0.50,
        decoder_cell_h: int = 2,
        decoder_cell_w: int = 2,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.dual_masking = bool(dual_masking)
        self.decoder_keep_ratio = float(decoder_keep_ratio)
        self.decoder_cell_h = int(decoder_cell_h)
        self.decoder_cell_w = int(decoder_cell_w)
        self.encoder = VideoMAEEncoder(
            num_frames=num_frames,
            img_size=img_size,
            tube_t=tube_t,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.decoder = VideoMAEDecoder(
            encoder_dim=embed_dim,
            num_tokens=self.encoder.num_tokens,
            tube_t=tube_t,
            patch_size=patch_size,
        )
        self.n_t = self.encoder.patch_embed.n_t
        self.n_h = self.encoder.patch_embed.n_h
        self.n_w = self.encoder.patch_embed.n_w

    def forward(
        self,
        x: torch.Tensor,
        ids_keep: torch.Tensor | None = None,
        ids_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:        (B, T, C, H, W)
            ids_keep: (B, n_visible) — if None, sample a fresh tube mask
            ids_mask: (B, n_masked)  — must be provided together with ids_keep
        Returns:
            pred:        (B, P, cube_dim)        — predictions at ids_predict
            ids_keep:    (B, n_visible)
            ids_mask:    (B, n_masked)           — encoder-masked positions
            ids_predict: (B, P)                  — positions ``pred`` covers
                                                   (V1: == ids_mask; V2: decoder-kept set)
        """
        if ids_keep is None:
            ids_keep, ids_mask = make_tube_mask(
                x.shape[0], self.n_t, self.n_h, self.n_w,
                self.mask_ratio, device=x.device,
            )
        ids_decoder_kept: torch.Tensor | None = None
        if self.dual_masking:
            ids_decoder_kept = make_running_cell_mask(
                x.shape[0], self.n_t, self.n_h, self.n_w,
                keep_ratio=self.decoder_keep_ratio,
                cell_h=self.decoder_cell_h,
                cell_w=self.decoder_cell_w,
                device=x.device,
            )
        visible = self.encoder(x, ids_keep=ids_keep)
        pred, ids_predict = self.decoder(visible, ids_keep, ids_mask, ids_decoder_kept)
        return pred, ids_keep, ids_mask, ids_predict


# ── SSL loss ───────────────────────────────────────────────────────────────────


def videomae_pixel_loss(
    pred: torch.Tensor,
    clips: torch.Tensor,
    ids_predict: torch.Tensor,
    tube_t: int = 2,
    patch_size: int = 16,
    norm_pix: bool = True,
    ids_encoder_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE reconstruction loss on per-cube normalized pixels.

    V1: pass ``ids_predict == ids_mask`` (decoder predicts at encoder-masked
    positions only); ``ids_encoder_mask`` is None. Loss is the mean MSE over
    all (B, P) prediction positions.

    V2 dual masking: ``ids_predict`` is the decoder-kept set (B, k) and
    ``ids_encoder_mask`` is the (B, n_masked) encoder-masked set. Loss is
    computed only at positions that are BOTH decoder-kept AND encoder-masked
    (the "invisible-only" objective from V2 Tab. 1).
    """
    target_full = patchify(clips, tube_t=tube_t, patch_size=patch_size)  # (B, N, cube_dim)
    B, N, _ = target_full.shape
    idx = ids_predict.unsqueeze(-1).expand(-1, -1, target_full.shape[-1])
    target = torch.gather(target_full, 1, idx)  # (B, P, cube_dim)

    if norm_pix:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True, unbiased=False)
        target = (target - mean) / (var + 1e-6).sqrt()

    sq = (pred - target.to(pred.dtype)) ** 2  # (B, P, cube_dim)

    if ids_encoder_mask is None:
        return sq.mean()

    # V2: per-prediction-position mean over cube_dim, then average only over
    # positions that are encoder-masked (the "invisible" subset).
    per_pos = sq.mean(dim=-1)  # (B, P)
    is_masked = torch.zeros(B, N, dtype=torch.bool, device=target.device)
    is_masked.scatter_(1, ids_encoder_mask, True)
    weight = torch.gather(is_masked, 1, ids_predict).to(per_pos.dtype)  # (B, P)
    return (per_pos * weight).sum() / weight.sum().clamp(min=1.0)


# ── Builder ────────────────────────────────────────────────────────────────────


@register_model("video_mae_vit")
def build_video_mae_vit(cfg: DictConfig) -> nn.Module:
    variant = str(cfg.model.get("variant", "vit_b"))
    if variant not in _VIT_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}. Choose from {sorted(_VIT_VARIANTS)}")
    arch = _VIT_VARIANTS[variant]
    return VideoMAEViT(
        num_classes=int(cfg.model.num_classes),
        num_frames=int(cfg.dataset.num_frames),
        img_size=int(cfg.dataset.get("image_size", 224)),
        tube_t=int(cfg.model.get("tube_t", 2)),
        patch_size=int(cfg.model.get("patch_size", 16)),
        embed_dim=arch["embed_dim"],
        depth=arch["depth"],
        num_heads=arch["num_heads"],
        mlp_ratio=float(cfg.model.get("mlp_ratio", 4.0)),
        drop_path_rate=float(cfg.model.get("drop_path_rate", 0.0)),
        dropout=float(cfg.model.get("dropout", 0.0)),
        head=str(cfg.model.get("head", "mean")),
        head_num_heads=int(cfg.model.get("head_num_heads", 4)),
        gradient_checkpointing=bool(cfg.model.get("gradient_checkpointing", False)),
        residual_variant=str(cfg.model.get("residual_variant", "prenorm")),
        hc_n=int(cfg.model.get("hc_n", 4)),
        hc_sk_iters=int(cfg.model.get("hc_sk_iters", 3)),
        hc_sk_tau=float(cfg.model.get("hc_sk_tau", 1.0)),
    )
