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
from omegaconf import DictConfig

from smth2smth.shared.models.registry import register_model

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


# ── Encoder ────────────────────────────────────────────────────────────────────


class VideoMAEEncoder(nn.Module):
    """ViT encoder with 3D cube embedding and learnable positional encoding.

    Accepts an optional ``ids_keep`` tensor to operate only on visible tokens
    (used during VideoMAE pretraining). Pass ``None`` for full-sequence mode
    (supervised fine-tuning / inference).
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
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed3D(num_frames, img_size, tube_t, patch_size, embed_dim)
        self.num_tokens = self.patch_embed.num_tokens

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))

        dpr = [drop_path_rate * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dpr[i])
            for i in range(depth)
        ])
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

        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)


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


# ── ViT variant table ──────────────────────────────────────────────────────────

_VIT_VARIANTS: dict[str, dict] = {
    "vit_s": dict(embed_dim=384,  depth=12, num_heads=6),
    "vit_b": dict(embed_dim=768,  depth=12, num_heads=12),
    "vit_l": dict(embed_dim=1024, depth=24, num_heads=16),
}


# ── SSL helpers ────────────────────────────────────────────────────────────────


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

    def __init__(self, encoder_dim: int, num_tokens: int) -> None:
        super().__init__()
        d = self.DECODER_DIM
        self.proj = nn.Linear(encoder_dim, d)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_tokens, d))
        dpr = [0.0] * self.DECODER_DEPTH
        self.blocks = nn.ModuleList([
            TransformerBlock(d, self.DECODER_HEADS, mlp_ratio=4.0, drop_path=dpr[i])
            for i in range(self.DECODER_DEPTH)
        ])
        self.norm = nn.LayerNorm(d)
        cube_dim = 3 * 2 * 16 * 16  # 1536; fixed for default tube_t=2, patch_size=16
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
    ) -> torch.Tensor:
        """
        Args:
            visible_tokens: (B, n_visible, encoder_dim)
            ids_keep:       (B, n_visible) token indices
            ids_mask:       (B, n_masked)  token indices
        Returns:
            (B, n_masked, cube_dim) — predictions at masked positions only
        """
        B = visible_tokens.shape[0]
        n_visible = ids_keep.shape[1]
        n_masked = ids_mask.shape[1]
        N = n_visible + n_masked

        tokens = self.proj(visible_tokens)  # (B, n_visible, d)

        # Build full-length sequence: scatter visible tokens + mask tokens.
        # Under AMP, ``tokens`` may be bfloat16 while ``mask_token`` is float32;
        # ``scatter_`` requires matching dtypes.
        full = self.mask_token.expand(B, N, -1).clone().to(dtype=tokens.dtype)
        idx_v = ids_keep.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        full.scatter_(1, idx_v, tokens)

        full = full + self.decoder_pos_embed.to(dtype=full.dtype)

        for block in self.blocks:
            full = block(full)
        full = self.norm(full)

        # Gather only the masked positions
        idx_m = ids_mask.unsqueeze(-1).expand(-1, -1, full.shape[-1])
        masked_out = torch.gather(full, 1, idx_m)   # (B, n_masked, d)
        return self.head(masked_out)                 # (B, n_masked, cube_dim)


# ── SSL pretrain model ─────────────────────────────────────────────────────────


class VideoMAEPretrainModel(nn.Module):
    """Encoder + decoder for VideoMAE masked-reconstruction pretraining."""

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
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
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
        )
        self.decoder = VideoMAEDecoder(
            encoder_dim=embed_dim,
            num_tokens=self.encoder.num_tokens,
        )
        self.n_t = self.encoder.patch_embed.n_t
        self.n_h = self.encoder.patch_embed.n_h
        self.n_w = self.encoder.patch_embed.n_w

    def forward(
        self,
        x: torch.Tensor,
        ids_keep: torch.Tensor | None = None,
        ids_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:        (B, T, C, H, W)
            ids_keep: (B, n_visible) — if None, sample a fresh tube mask
            ids_mask: (B, n_masked)  — must be provided together with ids_keep
        Returns:
            pred:     (B, n_masked, cube_dim)
            ids_keep: (B, n_visible)
            ids_mask: (B, n_masked)
        """
        if ids_keep is None:
            ids_keep, ids_mask = make_tube_mask(
                x.shape[0], self.n_t, self.n_h, self.n_w,
                self.mask_ratio, device=x.device,
            )
        visible = self.encoder(x, ids_keep=ids_keep)
        pred = self.decoder(visible, ids_keep, ids_mask)
        return pred, ids_keep, ids_mask


# ── SSL loss ───────────────────────────────────────────────────────────────────


def videomae_pixel_loss(
    pred: torch.Tensor,
    clips: torch.Tensor,
    ids_mask: torch.Tensor,
    tube_t: int = 2,
    patch_size: int = 16,
    norm_pix: bool = True,
) -> torch.Tensor:
    """MSE reconstruction loss on per-cube normalized pixels (masked positions).

    Args:
        pred:     (B, n_masked, cube_dim)
        clips:    (B, T, C, H, W)  — original input frames
        ids_mask: (B, n_masked)
        norm_pix: if True, normalize each cube to zero-mean unit-variance
    Returns:
        scalar loss
    """
    target = patchify(clips, tube_t=tube_t, patch_size=patch_size)  # (B, N, cube_dim)

    # Gather masked cubes
    idx = ids_mask.unsqueeze(-1).expand(-1, -1, target.shape[-1])
    target = torch.gather(target, 1, idx)  # (B, n_masked, cube_dim)

    if norm_pix:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True, unbiased=False)
        target = (target - mean) / (var + 1e-6).sqrt()

    return ((pred - target) ** 2).mean()


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
    )
