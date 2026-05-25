"""Training pipeline.

Run from the repo root::

    PYTHONPATH=src uv run python -m smth2smth.pipelines.train experiment=baseline_pretrained track=a

Tests can call :func:`run` directly with a hand-built ``DictConfig``; only the
:func:`main` wrapper depends on Hydra.
"""

from __future__ import annotations

import gc
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, WeightedRandomSampler

from smth2smth.shared.data import (
    VideoFrameDataset,
    build_time_reversal_table,
    build_track_a_temporal_reversal_map,
    build_transforms,
    collect_video_samples,
    describe_time_reversal_table,
    expand_train_samples_for_class_boosting,
)
from smth2smth.shared.engine import EpochStats, evaluate_epoch, train_one_epoch
from smth2smth.shared.io.checkpoints import load_checkpoint, save_checkpoint
from smth2smth.shared.models import build_model
from smth2smth.shared.models.video_mae import (
    VideoMAEViT,
    _VIT_VARIANTS,
    interpolate_pos_embed,
)
from smth2smth.shared.utils import (
    compute_class_weights,
    compute_sample_weights,
    set_seed,
    split_train_val,
    split_train_val_stratified,
)
from smth2smth.shared.utils.splits import label_counts
from smth2smth.shared.utils.wandb_run import (
    WandbTracker,
    build_step_metrics_callback,
    load_repo_dotenv,
    log_epoch_summary,
)

load_repo_dotenv()

CONFIGS_DIR = str(Path(__file__).resolve().parents[3] / "configs")


def _epoch_progress_stamp(epoch_one_indexed: int, total_epochs: int) -> str:
    """Prefix epoch logs with an ISO timestamp on a coarse grid (every 50 epochs)."""
    if (
        epoch_one_indexed == 1
        or epoch_one_indexed == total_epochs
        or epoch_one_indexed % 50 == 0
    ):
        return f"[{datetime.now().isoformat(timespec='seconds')}] "
    return ""


def _resolve_device(device_str: str) -> torch.device:
    """Resolve the requested device, falling back to CPU when CUDA is missing."""
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def _ssl_trunk_to_supervised_keys(trunk_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Adapt a plain-ResNet-50 SSL trunk state_dict to the supervised model's
    backbone state_dict layout.

    :func:`AvancedResNet50TSM` wraps every bottleneck block's ``conv1`` in a
    ``nn.Sequential(TemporalShift, conv1)``, which renames the parameter key
    from ``layerN.M.conv1.weight`` to ``layerN.M.conv1.1.weight``. SSL
    pretraining has no temporal axis so its trunk has plain ``conv1``; this
    helper performs the renaming so the SSL weights drop into the supervised
    model's ``backbone`` directly. Other keys (``conv2``, ``conv3``, batchnorms,
    stem, etc.) are passed through unchanged.

    The returned dict is also prefixed with ``backbone.`` so it can be passed
    straight to ``AvancedResNet50TSM.load_state_dict(..., strict=False)``.
    """
    import re

    block_conv1_re = re.compile(r"^(layer[1-4]\.\d+\.conv1)\.(weight|bias)$")
    out: dict[str, torch.Tensor] = {}
    for k, v in trunk_state.items():
        match = block_conv1_re.match(k)
        if match is not None:
            new_k = f"{match.group(1)}.1.{match.group(2)}"
        else:
            new_k = k
        out[f"backbone.{new_k}"] = v
    return out


def _balanced_per_class_subsample(
    samples: list[tuple[Path, int]],
    *,
    max_per_class: int,
    seed: int,
) -> list[tuple[Path, int]]:
    """Keep at most ``max_per_class`` samples for each integer class label.

    Selection is deterministic given ``seed``: we draw a permutation of the
    indices within each class and take the first ``max_per_class``. The
    returned list preserves the original ``samples`` ordering for the kept
    indices, which keeps downstream sort/shuffle behaviour unchanged.

    Args:
        samples: List of ``(video_dir, class_index)`` pairs.
        max_per_class: Maximum clips kept per class label. Non-positive
            values yield an empty subset.
        seed: NumPy seed for the per-class permutation.

    Returns:
        A new list with at most ``max_per_class`` samples per label, in
        the same order as ``samples`` for the surviving indices.
    """
    if max_per_class <= 0:
        return []
    import numpy as np

    rng = np.random.default_rng(int(seed))
    indices_by_label: dict[int, list[int]] = {}
    for idx, (_, label) in enumerate(samples):
        indices_by_label.setdefault(int(label), []).append(idx)

    kept: set[int] = set()
    for label_indices in indices_by_label.values():
        if len(label_indices) <= max_per_class:
            kept.update(label_indices)
            continue
        perm = rng.permutation(len(label_indices))[:max_per_class]
        for i in perm:
            kept.add(label_indices[int(i)])
    return [samples[i] for i in range(len(samples)) if i in kept]


def _collect_extra_train_samples(
    use_extra: bool,
    train_extra_dir: str | Path | None,
) -> list[tuple[Path, int]]:
    """Collect open-world extra training clips (Track B / E6).

    Returns an empty list when ``use_extra`` is false. When enabled, every clip
    under ``train_extra_dir`` is collected with the same class-prefix parsing as
    the primary train split, so the extra rows can be appended to (and only to)
    the training set. Fails loudly when ``use_extra`` is true but the directory
    is unset or missing — silently training on zero extra clips would corrupt an
    ablation row.

    Args:
        use_extra: ``dataset.use_extra`` flag.
        train_extra_dir: ``dataset.train_extra_dir`` path.

    Returns:
        ``(video_dir, class_index)`` pairs from ``train_extra_dir`` (possibly
        empty when ``use_extra`` is false).

    Raises:
        ValueError: If ``use_extra`` is true but ``train_extra_dir`` is unset.
        FileNotFoundError: If ``use_extra`` is true but the directory is absent.
    """
    if not use_extra:
        return []
    if not train_extra_dir:
        raise ValueError(
            "dataset.use_extra=true requires dataset.train_extra_dir to be set."
        )
    extra_dir = Path(str(train_extra_dir)).resolve()
    if not extra_dir.is_dir():
        raise FileNotFoundError(
            f"dataset.use_extra=true but train_extra_dir does not exist: {extra_dir}. "
            "Run scripts/download_ssv2_subset_4frame.py first, or set use_extra=false."
        )
    return collect_video_samples(extra_dir)


def _split_trainable_params_head_vs_lora(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Partition trainable parameters into probe/head vs PEFT LoRA adapters.

    HuggingFace PEFT names all adapter weights with a ``lora_`` prefix in the
    parameter name: ``lora_A`` / ``lora_B`` for vanilla LoRA, plus
    ``lora_magnitude_vector`` for DoRA. We route every ``lora_`` param to the
    adapter group so DoRA's magnitude vectors get the dedicated LoRA LR rather
    than the (typically higher) head LR. Everything else trainable is the head.

    Args:
        model: Network possibly wrapped with ``PeftModel`` on the backbone.

    Returns:
        ``(head_params, lora_params)`` lists; either list may be empty.
    """
    head_params: list[nn.Parameter] = []
    lora_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(param)
        else:
            head_params.append(param)
    return head_params, lora_params


def _split_dual_stream_backbone_params(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
    """Partition trainable params of ``DualStreamRgbDiffTSM`` into 3 groups.

    The dual-stream model fuses a ResNet-50 RGB branch (``rgb_backbone.*``)
    and a ResNet-34 motion branch (``motion_backbone.*``); all remaining
    trainable tensors are the fusion projection, optional attention pool,
    dropout, and the linear classifier. We expose the three groups so the
    caller can give the smaller motion backbone a different learning rate
    than the (much larger) RGB backbone.

    Args:
        model: Instance of ``DualStreamRgbDiffTSM`` (or any module that
            exposes ``rgb_backbone`` / ``motion_backbone`` submodules).

    Returns:
        Tuple ``(rgb_params, motion_params, rest_params)``; any of the
        three lists may be empty (e.g. if a backbone is fully frozen).
    """
    rgb_params: list[nn.Parameter] = []
    motion_params: list[nn.Parameter] = []
    rest_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("rgb_backbone."):
            rgb_params.append(param)
        elif name.startswith("motion_backbone."):
            motion_params.append(param)
        else:
            rest_params.append(param)
    return rgb_params, motion_params, rest_params


def _is_hc_scalar(name: str) -> bool:
    """Detect Hyper-Connection routing parameters in a VideoMAEViT param name.

    HC scalars (``M`` / ``M_raw`` / ``alpha_pre`` / ``beta`` per sublayer plus
    a top-level ``alpha_out``) are excluded from weight decay and from
    layer-wise LR decay — treated like LayerNorm gains, per the HC paper's
    recipe and the experiment doc's hard rules. The detection is purely
    name-based so the LLRD builder can stay model-agnostic.
    """
    if name == "encoder.alpha_out":
        return True
    if not name.startswith("encoder.blocks."):
        return False
    suffix = name.split(".", 3)[3] if name.count(".") >= 3 else ""
    if not suffix.startswith(("attn_router.", "mlp_router.")):
        return False
    leaf = suffix.split(".", 1)[1]
    return leaf in {"M", "M_raw", "alpha_pre", "beta"}


_TEMPORAL_LEAF_PREFIXES = (
    "norm_t.",
    "temporal_attn.",
    "temporal_fc.",
    "t_adapter.",
    "joint_adapter.",
)


def _is_new_temporal_param(name: str) -> bool:
    """Detect Arch-3/Arch-4 temporal-module params living inside an encoder block.

    The TimeSformer-style temporal attention + ``temporal_fc`` (divided_st) and
    the AIM zero-init adapters + their ``norm_t`` (aim_reuse) are *randomly /
    zero* initialised. The experiment doc's single highest-leverage recipe rule
    is that these new modules train at the full base LR, NOT the LLRD-decayed
    rate of the pretrained block they are nested in ("randomly-initialized
    modules get base LR; pretrained layers get LLRD"). Detection is name-based
    so the LLRD builder stays model-agnostic, mirroring :func:`_is_hc_scalar`.
    """
    if not name.startswith("encoder.blocks."):
        return False
    parts = name.split(".", 3)
    if len(parts) < 4:
        return False
    return parts[3].startswith(_TEMPORAL_LEAF_PREFIXES)


def _videomae_layer_id(name: str, depth: int) -> int:
    """Map a ``VideoMAEViT`` parameter name to a depth index for LLRD.

    Layer 0 is the patch/positional embedding (lowest, most decayed LR);
    encoder block ``i`` is layer ``i + 1``; the final encoder LayerNorm, the
    attentive-pool head, and the classifier are the top layer ``depth + 1``
    (full base LR). Mirrors the BEiT / MAE ``get_num_layer`` convention.

    HC scalars (``encoder.alpha_out`` and per-block ``attn_router`` /
    ``mlp_router`` parameters) are routed to the top layer so they train at
    the full base LR — see :func:`_is_hc_scalar`. The same applies to the new
    temporal modules (Arch 3/4) nested inside pretrained blocks — see
    :func:`_is_new_temporal_param`.
    """
    if _is_hc_scalar(name) or _is_new_temporal_param(name):
        return depth + 1
    if name.startswith("encoder.patch_embed") or name == "encoder.pos_embed":
        return 0
    if name.startswith("encoder.blocks."):
        try:
            return int(name.split(".")[2]) + 1
        except (IndexError, ValueError):
            return depth + 1
    # encoder.norm, attn_pool.*, classifier.* (and any future head params).
    return depth + 1


def _build_llrd_param_groups(
    model: nn.Module,
    *,
    base_lr: float,
    weight_decay: float,
    layer_decay: float,
    depth: int,
) -> list[dict[str, Any]]:
    """Layer-wise LR decay param groups for ``video_mae_vit`` fine-tuning.

    Layer ``l`` is scaled by ``layer_decay ** (depth + 1 - l)`` so the head
    trains at ``base_lr`` and the patch embedding at the most decayed rate.
    Bias and 1-D (norm) parameters are excluded from weight decay, the
    standard ViT fine-tuning recipe. Only trainable parameters are included,
    so this composes with ``model.freeze_backbone``.
    """
    n_top = depth + 1
    groups: dict[tuple[int, bool], dict[str, Any]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_id = _videomae_layer_id(name, depth)
        # HC matrices are 2-D (n, n) so ``ndim<=1`` would not catch them, but
        # the doc treats them like LN gains — force ``no_decay`` for any
        # HC routing scalar.
        no_decay = (
            param.ndim <= 1
            or name.endswith(".bias")
            or _is_hc_scalar(name)
        )
        key = (layer_id, no_decay)
        if key not in groups:
            scale = layer_decay ** (n_top - layer_id)
            groups[key] = {
                "params": [],
                "lr": base_lr * scale,
                "weight_decay": 0.0 if no_decay else weight_decay,
                "warmup_lr_max": base_lr * scale,
                "name": f"llrd_l{layer_id}{'_nd' if no_decay else ''}",
            }
        groups[key]["params"].append(param)
    if not groups:
        raise RuntimeError(
            "LLRD produced 0 trainable groups; check model.freeze_backbone "
            "or that the model is video_mae_vit."
        )
    return [groups[k] for k in sorted(groups)]


def _is_new_module_param(name: str) -> bool:
    """Random-init head + temporal modules for stabilized fine-tuning (Round 2).

    These parameters use ``training.new_module_lr`` and a longer warmup instead
    of sharing the full base LR with the LLRD top group.
    """
    return (
        name.startswith("pool_head.")
        or name.startswith("classifier.")
        or _is_new_temporal_param(name)
    )


def _backbone_llrd_layer_id(name: str, depth: int) -> int:
    """Layer index for pretrained backbone tensors only (excludes pool_head)."""
    if _is_hc_scalar(name):
        return depth + 1
    if name.startswith("encoder.patch_embed") or name == "encoder.pos_embed":
        return 0
    if name.startswith("encoder.blocks."):
        try:
            return int(name.split(".")[2]) + 1
        except (IndexError, ValueError):
            return depth + 1
    if name.startswith("encoder.norm"):
        return depth + 1
    return depth + 1


def _build_llrd_stabilized_param_groups(
    model: nn.Module,
    *,
    base_lr: float,
    new_module_lr: float,
    weight_decay: float,
    layer_decay: float,
    depth: int,
    backbone_warmup_epochs: int,
    new_module_warmup_epochs: int,
) -> list[dict[str, Any]]:
    """LLRD on the pretrained backbone; lower LR + longer warmup on new modules."""
    n_top = depth + 1
    groups: dict[tuple[str, int, bool], dict[str, Any]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        no_decay = (
            param.ndim <= 1
            or name.endswith(".bias")
            or _is_hc_scalar(name)
        )
        if _is_new_module_param(name):
            kind = "new"
            layer_key = 0
            lr = new_module_lr
            warm = new_module_warmup_epochs
            gname = f"new_module{'_nd' if no_decay else ''}"
        else:
            kind = "bb"
            layer_key = _backbone_llrd_layer_id(name, depth)
            lr = base_lr * (layer_decay ** (n_top - layer_key))
            warm = backbone_warmup_epochs
            gname = f"llrd_l{layer_key}{'_nd' if no_decay else ''}"
        key = (kind, layer_key, no_decay)
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": lr,
                "weight_decay": 0.0 if no_decay else weight_decay,
                "warmup_lr_max": lr,
                "warmup_epochs": warm,
                "name": gname,
            }
        groups[key]["params"].append(param)
    if not groups:
        raise RuntimeError(
            "Stabilized LLRD produced 0 trainable groups; check freeze_backbone."
        )
    return [groups[k] for k in sorted(groups, key=lambda x: (x[0], x[1], x[2]))]


def _new_module_param_ids(param_groups: list[dict[str, Any]]) -> set[int]:
    """Parameter ids in optimizer groups tagged ``new_module*``."""
    out: set[int] = set()
    for g in param_groups:
        if str(g.get("name", "")).startswith("new_module"):
            for p in g["params"]:
                out.add(id(p))
    return out


def _log_diverse_arch_init_diagnostics(model: nn.Module) -> None:
    """One-time init-time sanity print for the diverse-head / temporal archs.

    For temporal models (Arch 3/4) this is the doc's "single highest-payoff
    sanity check": every temporal identity projection (``temporal_fc`` /
    adapter up-projection) must be *exactly* zero after init_from loading, so
    the temporal path contributes nothing at step 0 and the encoder output
    equals the plain pretrained backbone. A non-zero value means the identity
    init was clobbered (e.g. by a stray re-init) and the run would train the
    temporal blocks from scratch — the doc's top failure mode for Arch 3.

    For the Perceiver head (Arch 2) it prints the init query pairwise cosine
    (should be ≈0; >0.7 later means query collapse).
    """
    from smth2smth.shared.models.video_mae import (
        AIMReuseBlock,
        DividedSpaceTimeBlock,
        VideoMAEViT,
    )

    if not isinstance(model, VideoMAEViT):
        return
    enc = model.encoder
    temporal_mode = getattr(enc, "temporal_mode", "none")
    if temporal_mode != "none":
        max_abs = 0.0
        n_blocks = 0
        for blk in enc.blocks:
            if isinstance(blk, DividedSpaceTimeBlock):
                n_blocks += 1
                max_abs = max(
                    max_abs,
                    float(blk.temporal_fc.weight.abs().max()),
                    float(blk.temporal_fc.bias.abs().max()),
                )
            elif isinstance(blk, AIMReuseBlock):
                n_blocks += 1
                for adapter in (blk.t_adapter, blk.joint_adapter):
                    max_abs = max(
                        max_abs,
                        float(adapter.up.weight.abs().max()),
                        float(adapter.up.bias.abs().max()),
                    )
        if max_abs == 0.0:
            status = "OK (temporal path == identity at step 0)"
        else:
            status = f"BROKEN: identity projection not zero (max|param|={max_abs:.3e})"
        print(
            f"[diverse-arch] temporal_mode={temporal_mode}, {n_blocks} temporal "
            f"block(s) at full base LR; identity-at-init {status}"
        )
    pool_head = getattr(model, "pool_head", None)
    if pool_head is not None:
        print(
            f"[diverse-arch] pool_head: num_queries={pool_head.num_queries}, "
            f"init query pairwise cosine={pool_head.query_pairwise_cosine():.4f} "
            f"(>0.7 later ⇒ query collapse)."
        )


class RepeatedAugSampler(torch.utils.data.Sampler[int]):
    """DeiT-style repeated augmentation: each clip appears ``repeats`` times.

    The dataset transform is stochastic per ``__getitem__``, so consecutive
    repeats of the same index yield differently-augmented views. The epoch
    length is kept at ``len(dataset)`` (so wall-clock per epoch is unchanged;
    the number of *unique* clips per epoch is ``len(dataset) // repeats``),
    matching the VideoMAE FT recipe where ``batch_size`` is halved to keep the
    effective view-batch constant. ``repeats=1`` is a plain shuffled pass.
    """

    def __init__(self, num_samples: int, repeats: int = 1, shuffle: bool = True) -> None:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}.")
        if repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {repeats}.")
        self.num_samples = int(num_samples)
        self.repeats = int(repeats)
        self.shuffle = bool(shuffle)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        if self.shuffle:
            order = torch.randperm(self.num_samples).tolist()
        else:
            order = list(range(self.num_samples))
        out: list[int] = []
        for idx in order:
            out.extend([idx] * self.repeats)
            if len(out) >= self.num_samples:
                break
        return iter(out[: self.num_samples])


def _free_cuda_memory(reason: str = "") -> None:
    """Release Python references and empty the CUDA caching allocator.

    Important for Hydra multirun, where N back-to-back jobs share a single
    Python process: PyTorch holds GPU memory in its caching allocator after a
    job finishes, so the next job can hit OOM even though the previous model
    has gone out of scope. Calling this between jobs (and after an OOM) keeps
    the GPU clean.

    Args:
        reason: Optional string included in the log line for traceability.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        suffix = f" ({reason})" if reason else ""
        free, total = torch.cuda.mem_get_info()
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        print(f"[cuda] cache emptied{suffix}. Free: {free_gb:.2f} / {total_gb:.2f} GiB")


def run(cfg: DictConfig) -> Path | None:
    """Train a model and write the best-by-val-top1 checkpoint.

    Args:
        cfg: Composed Hydra configuration.

    Returns:
        Path to the saved checkpoint, or ``None`` if training never improved
        the initial accuracy (no checkpoint written).
    """
    print(OmegaConf.to_yaml(cfg))

    _free_cuda_memory(reason="run-start")

    set_seed(int(cfg.seed))
    device = _resolve_device(str(cfg.training.device))

    train_dir = Path(cfg.dataset.train_dir).resolve()
    all_samples = collect_video_samples(train_dir)

    max_samples_per_class = cfg.dataset.get("max_samples_per_class")
    if max_samples_per_class is not None:
        all_samples = _balanced_per_class_subsample(
            all_samples,
            max_per_class=int(max_samples_per_class),
            seed=int(cfg.dataset.seed),
        )
        print(
            f"[data] balanced subsample: kept {len(all_samples)} clips "
            f"(at most {int(max_samples_per_class)} per class, seed={int(cfg.dataset.seed)})"
        )

    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        all_samples = all_samples[: int(max_samples)]

    use_official_val = bool(cfg.dataset.get("use_official_val", False))
    include_val_in_train = bool(cfg.dataset.get("include_val_in_train", False))
    val_holdout_ratio = float(cfg.dataset.get("official_val_holdout_ratio", 0.0) or 0.0)
    use_val_holdout = use_official_val and val_holdout_ratio > 0.0
    val_samples_all: list[tuple[Path, int]] | None = None
    if use_official_val:
        # Validate on the official held-out folder. The internal 80/20 split is
        # bypassed: training uses *all* of ``train_dir``, validation uses
        # *all* of ``val_dir``. Optionally also train on ``val_dir`` clips.
        val_dir_for_val = Path(cfg.dataset.val_dir).resolve()
        val_samples_all = collect_video_samples(val_dir_for_val)
        if max_samples is not None:
            val_samples_all = val_samples_all[: int(max_samples)]
        train_samples = list(all_samples)
        train_sources = f"train_dir={len(all_samples)}"
        if val_holdout_ratio > 0.0:
            if include_val_in_train:
                print(
                    "[data] official_val_holdout_ratio>0: ignoring "
                    "include_val_in_train (holdout split adds val to train)."
                )
            val_for_train, val_samples = split_train_val_stratified(
                val_samples_all,
                val_ratio=val_holdout_ratio,
                seed=int(cfg.dataset.seed),
            )
            train_samples.extend(val_for_train)
            train_sources += (
                f" + val_dir={len(val_for_train)} "
                f"({1.0 - val_holdout_ratio:.0%} stratified)"
            )
            full_counts = label_counts(val_samples_all)
            hold_counts = label_counts(val_samples)
            n_classes_full = len(full_counts)
            n_classes_hold = len(hold_counts)
            min_hold = min(hold_counts.values()) if hold_counts else 0
            max_hold = max(hold_counts.values()) if hold_counts else 0
            print(
                f"[data] use_official_val=true, official_val_holdout_ratio="
                f"{val_holdout_ratio:g} (stratified per class): "
                f"train={len(train_samples)} ({train_sources}), "
                f"val_holdout={len(val_samples)} / val_total={len(val_samples_all)} "
                f"(from {val_dir_for_val})"
            )
            print(
                f"[data] holdout classes: {n_classes_hold}/{n_classes_full}; "
                f"clips per class in holdout min={min_hold} max={max_hold}"
            )
        elif include_val_in_train:
            val_samples = val_samples_all
            train_samples.extend(val_samples)
            train_sources += f" + val_dir={len(val_samples)}"
            print(
                f"[data] use_official_val=true: train={len(train_samples)} ({train_sources}), "
                f"val={len(val_samples)} (from {val_dir_for_val})"
            )
        else:
            val_samples = val_samples_all
            print(
                f"[data] use_official_val=true: train={len(train_samples)} ({train_sources}), "
                f"val={len(val_samples)} (from {val_dir_for_val})"
            )
    else:
        train_samples, val_samples = split_train_val(
            all_samples,
            val_ratio=float(cfg.dataset.val_ratio),
            seed=int(cfg.dataset.seed),
        )

    # Open-world (Track B / E6) extra training data. Appended to the *training*
    # set only; validation always stays on the in-distribution split above.
    extra_samples = _collect_extra_train_samples(
        bool(cfg.dataset.get("use_extra", False)),
        cfg.dataset.get("train_extra_dir"),
    )
    if extra_samples:
        train_samples = list(train_samples) + extra_samples
        print(
            f"[data] use_extra=true: +{len(extra_samples)} extra train clips from "
            f"{Path(str(cfg.dataset.get('train_extra_dir'))).resolve()} "
            f"(train total={len(train_samples)})"
        )

    # Class indices that actually have at least one training sample. Recorded
    # in the checkpoint so :mod:`smth2smth.pipelines.submit` can mask logits
    # for never-trained classes (e.g. the missing class 27 in the Track A
    # dataset would otherwise win a fraction of test predictions by accident).
    trained_class_indices: list[int] = sorted({int(label) for _, label in train_samples})

    class_boost_cfg = cfg.dataset.get("class_boosting")
    class_boosting_enabled = class_boost_cfg is not None and bool(class_boost_cfg.get("enabled", False))
    if class_boosting_enabled:
        pair_for_boost = build_track_a_temporal_reversal_map()
        n_disk_rows = len(train_samples)
        train_samples = expand_train_samples_for_class_boosting(train_samples, pair_for_boost)
        print(
            f"[data] class_boosting: expanded train rows {n_disk_rows} -> {len(train_samples)} "
            f"(deterministic paired-verb duplicates)"
        )

    use_imagenet_norm = bool(cfg.model.get("pretrained", False)) if hasattr(cfg.model, "get") else bool(cfg.model.pretrained)
    augment_cfg = cfg.get("augment") if hasattr(cfg, "get") else None
    train_transform = build_transforms(
        image_size=int(cfg.dataset.image_size),
        is_training=True,
        use_imagenet_norm=use_imagenet_norm,
        augment=augment_cfg,
    )
    eval_transform = build_transforms(
        image_size=int(cfg.dataset.image_size),
        is_training=False,
        use_imagenet_norm=use_imagenet_norm,
        augment=augment_cfg,
    )

    num_frames = int(cfg.dataset.num_frames)

    # Time-reversal augmentation (label-aware). Default ``time_reversal_prob=0``
    # is a no-op. When > 0, we build a per-class remap from the class folder
    # names and pass it to the train dataset only; the val dataset never
    # reverses (we want unbiased val numbers).
    time_reversal_prob = float(cfg.dataset.get("time_reversal_prob", 0.0))
    tr_perm: torch.Tensor | None = None
    tr_allow: torch.Tensor | None = None
    if time_reversal_prob > 0.0:
        tr_perm, tr_allow = build_time_reversal_table(
            train_dir=train_dir, num_classes=int(cfg.num_classes)
        )
        print(f"[time-reversal] prob={time_reversal_prob}. " + describe_time_reversal_table(tr_perm, tr_allow))

    tr_aug_cfg = cfg.dataset.get("temporal_reversal_augment")
    temporal_reversal_map = None
    temporal_reversal_prob = 0.0
    if class_boosting_enabled and tr_aug_cfg is not None and bool(tr_aug_cfg.get("enabled", False)):
        print(
            "[data] temporal_reversal_augment disabled while class_boosting is enabled "
            "(paired augmentation is already deterministic)."
        )
    elif tr_aug_cfg is not None and bool(tr_aug_cfg.get("enabled", False)):
        temporal_reversal_map = build_track_a_temporal_reversal_map()
        temporal_reversal_prob = float(tr_aug_cfg.get("prob", 0.5))
        print(
            f"[data] temporal_reversal_augment: prob={temporal_reversal_prob}, "
            f"paired_labels={len(temporal_reversal_map)}"
        )

    train_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=num_frames,
        transform=train_transform,
        sample_list=train_samples,
        time_reversal_prob=time_reversal_prob,
        time_reversal_perm=tr_perm,
        time_reversal_allow_mask=tr_allow,
        temporal_reversal_pair_to_opposite=temporal_reversal_map,
        temporal_reversal_prob=temporal_reversal_prob,
    )
    val_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=val_samples,
    )

    # Class-balanced sampler (opt-in). When the policy is ``"none"`` we keep
    # ``shuffle=True`` so the legacy run path is byte-for-byte identical.
    cb_sampler_policy = str(cfg.training.get("class_balance_sampler", "none")).lower()
    sampler: WeightedRandomSampler | None = None
    if cb_sampler_policy != "none":
        sample_weights = compute_sample_weights(
            samples=train_samples,
            num_classes=int(cfg.num_classes),
            policy=cb_sampler_policy,
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(train_samples),
            replacement=True,
        )
        n_train_by_class = sorted({int(s[1]) for s in train_samples})
        print(
            f"[class-balance] sampler={cb_sampler_policy!r}; "
            f"trained classes={len(n_train_by_class)}, "
            f"effective per-class draw equalised by {cb_sampler_policy}."
        )

    pin_memory = device.type == "cuda"
    repeated_aug = max(1, int(cfg.training.get("repeated_aug", 1)))
    train_sampler: torch.utils.data.Sampler[int] | WeightedRandomSampler | None = sampler
    if repeated_aug > 1:
        if sampler is not None:
            print(
                "[data] repeated_aug ignored when class_balance_sampler is active "
                f"(policy={cb_sampler_policy!r})."
            )
        else:
            train_sampler = RepeatedAugSampler(
                len(train_samples), repeats=repeated_aug, shuffle=True
            )
            print(
                f"[data] repeated_aug={repeated_aug}: each clip drawn up to "
                f"{repeated_aug}x per epoch with independent augmentations."
            )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=int(cfg.training.num_workers),
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=pin_memory,
    )
    val_honest_loader: DataLoader | None = None
    if use_val_holdout and val_samples_all is not None:
        val_honest_dataset = VideoFrameDataset(
            root_dir=train_dir,
            num_frames=num_frames,
            transform=eval_transform,
            sample_list=val_samples_all,
        )
        val_honest_loader = DataLoader(
            val_honest_dataset,
            batch_size=int(cfg.training.batch_size),
            shuffle=False,
            num_workers=int(cfg.training.num_workers),
            pin_memory=pin_memory,
        )
        print(
            f"[data] dual val eval each epoch: holdout n={len(val_samples)} "
            f"(checkpoint/early-stop), honest n={len(val_samples_all)} (full official val; "
            f"includes clips also used in training)"
        )

    model = build_model(cfg).to(device)

    if bool(cfg.model.get("freeze_backbone", False)):
        if isinstance(model, VideoMAEViT):
            model.freeze_backbone_for_classifier_tune()
            print("[model] freeze_backbone=true: encoder + attentive pool frozen; classifier only.")
        else:
            print("[model] freeze_backbone ignored (only supported for video_mae_vit).")

    # Optional: warm-start the trunk from a self-supervised checkpoint produced
    # by ``pretrain_ssl.py``. Track-A safe: the SSL trunk was trained from
    # random init on the *provided* unlabeled frames only, so loading it here
    # introduces no external knowledge. The hook is opt-in (``model.init_from``
    # null by default) and the load is non-strict so slight architectural
    # mismatches (e.g. extra BN buffers) don't fail the supervised run.
    init_from = cfg.model.get("init_from") if hasattr(cfg.model, "get") else None
    if init_from:
        if str(cfg.model.name) == "dual_stream_rgb_diff_tsm":
            print(
                "[init_from] skipped for dual_stream_rgb_diff_tsm: SSL checkpoints target "
                "AvancedResNet50TSM backbone keys only (dual-stream uses rgb_backbone / motion_backbone)."
            )
        else:
            init_path = Path(str(init_from)).resolve()
            if not init_path.is_file():
                raise SystemExit(f"model.init_from points to a missing file: {init_path}")
            payload = torch.load(init_path, map_location=device, weights_only=False)
            trunk_state = payload.get("trunk_state_dict") if isinstance(payload, dict) else None
            if trunk_state is None:
                raise SystemExit(
                    f"{init_path} does not contain a 'trunk_state_dict' key; "
                    "expected an SSL checkpoint produced by pretrain_ssl.py."
                )
            model_name = str(cfg.model.name)
            if model_name == "video_mae_vit":
                # VideoMAE checkpoints store keys with ``encoder.`` prefix
                # already present (saved by pretrain_videomae.py). Load
                # directly without the ResNet-specific key renaming.
                if bool(cfg.model.get("interpolate_pos_embed", False)):
                    pe_key = "encoder.pos_embed"
                    if pe_key in trunk_state:
                        pe = trunk_state[pe_key]
                        src_frames = int(
                            cfg.model.get("interpolate_src_num_frames", num_frames)
                        )
                        src_size = int(cfg.model.get("interpolate_src_image_size", 224))
                        dst_size = int(cfg.dataset.image_size)
                        if pe.shape[1] != model.encoder.num_tokens:
                            trunk_state[pe_key] = interpolate_pos_embed(
                                pe,
                                src_num_frames=src_frames,
                                src_img_size=src_size,
                                dst_num_frames=num_frames,
                                dst_img_size=dst_size,
                                tube_t=int(cfg.model.get("tube_t", 2)),
                                patch_size=int(cfg.model.get("patch_size", 16)),
                            )
                            print(
                                f"[init_from] interpolated pos_embed "
                                f"{pe.shape[1]} -> {trunk_state[pe_key].shape[1]} tokens "
                                f"({src_size}->{dst_size}, T={src_frames}->{num_frames})."
                            )
                encoder_keys = {k for k in trunk_state if k.startswith("encoder.")}
                missing, unexpected = model.load_state_dict(trunk_state, strict=False)
                encoder_missing = [k for k in missing if k.startswith("encoder.")]
                print(
                    f"[init_from] loaded {len(encoder_keys)} encoder tensors from {init_path}. "
                    f"encoder-missing={len(encoder_missing)}, unexpected={len(unexpected)}"
                )
                if encoder_missing:
                    print(f"[init_from] encoder keys NOT covered by SSL: {encoder_missing[:8]}...")
            else:
                prefixed = _ssl_trunk_to_supervised_keys(trunk_state)
                missing, unexpected = model.load_state_dict(prefixed, strict=False)
                # ``missing`` will include classifier / attn_pool keys -- expected.
                backbone_missing = [k for k in missing if k.startswith("backbone.")]
                print(
                    f"[init_from] loaded {len(prefixed)} trunk tensors from {init_path}. "
                    f"backbone-missing={len(backbone_missing)}, unexpected={len(unexpected)}"
                )
                if backbone_missing:
                    print(f"[init_from] backbone keys NOT covered by SSL: {backbone_missing[:8]}...")

    # Diverse-head / temporal-arch init-time sanity print (Arch 1–4). No-op for
    # the mean-pool control. Runs after init_from so the identity-at-init check
    # reflects the loaded weights.
    _log_diverse_arch_init_diagnostics(model)

    # Class-balanced cross-entropy weights (opt-in). Composes with
    # label-smoothing and video-mixing: the same tensor is passed both to
    # ``nn.CrossEntropyLoss(weight=...)`` (plain-CE path) and to the
    # soft-target CE used by the mixing augmentations.
    cb_loss_policy = str(cfg.training.get("class_balance_loss", "none")).lower()
    cb_loss_beta = float(cfg.training.get("class_balance_beta", 0.999))
    class_weights = compute_class_weights(
        samples=train_samples,
        num_classes=int(cfg.num_classes),
        policy=cb_loss_policy,
        beta=cb_loss_beta,
    )
    if class_weights is not None:
        class_weights = class_weights.to(device)
        masked = (class_weights == 0).sum().item()
        print(
            f"[class-balance] loss policy={cb_loss_policy!r}, beta={cb_loss_beta}; "
            f"min={float(class_weights[class_weights > 0].min()):.3f}, "
            f"max={float(class_weights.max()):.3f}, "
            f"never-trained classes set to 0: {int(masked)}."
        )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer_name = str(cfg.training.get("optimizer", "adam")).lower()
    head_lr = float(cfg.training.lr)
    weight_decay = float(cfg.training.get("weight_decay", 0.0))
    lora_enabled = bool(cfg.model.get("lora_enabled", False)) if hasattr(cfg, "model") else False

    # Dual-stream (RGB ResNet-50 + motion ResNet-34) per-backbone LR. When
    # either ``training.motion_lr`` (absolute) or ``training.motion_lr_ratio``
    # (relative to ``training.lr``) is set AND the model is the dual-stream
    # builder, we construct three optimizer groups (rgb_backbone /
    # motion_backbone / rest=fuse+head+classifier) so the smaller ResNet-34
    # branch can train at a slower / faster rate than the ResNet-50 branch.
    # Mutually exclusive with the LoRA two-group path; LoRA is not supported
    # by the dual-stream builder, so this is well-defined.
    is_dual_stream = (
        hasattr(cfg, "model")
        and str(cfg.model.get("name", "")) == "dual_stream_rgb_diff_tsm"
    )
    motion_lr_ratio_raw = cfg.training.get("motion_lr_ratio") if hasattr(cfg, "training") else None
    motion_lr_raw = cfg.training.get("motion_lr") if hasattr(cfg, "training") else None
    use_dual_stream_groups = is_dual_stream and (
        motion_lr_ratio_raw is not None or motion_lr_raw is not None
    )

    head_params, lora_params = _split_trainable_params_head_vs_lora(model)
    if not head_params and not lora_params:
        raise RuntimeError(
            "No trainable parameters found in the model; cannot construct an optimizer."
        )
    lora_lr_ratio = float(cfg.training.get("lora_lr_ratio", 0.25))
    lora_lr_raw = cfg.training.get("lora_lr")
    if lora_lr_raw is not None:
        eff_lora_lr = float(lora_lr_raw)
    else:
        eff_lora_lr = head_lr * lora_lr_ratio

    use_lora_group = (
        lora_enabled
        and bool(head_params)
        and bool(lora_params)
        and not use_dual_stream_groups
    )
    if lora_enabled and not lora_params:
        print(
            "[optim] model.lora_enabled but no trainable lora_A/lora_B tensors; "
            "using a single LR group."
        )

    warmup_epochs = int(cfg.training.get("warmup_epochs", 0))
    layer_decay_raw = cfg.training.get("layer_decay")
    model_name_for_llrd = str(cfg.model.get("name", "")) if hasattr(cfg, "model") else ""
    use_llrd = (
        layer_decay_raw is not None
        and model_name_for_llrd == "video_mae_vit"
        and not use_dual_stream_groups
        and not use_lora_group
    )
    new_module_lr_raw = cfg.training.get("new_module_lr") if hasattr(cfg, "training") else None
    new_module_warmup_epochs = int(
        cfg.training.get("new_module_warmup_epochs", 10) if hasattr(cfg, "training") else 10
    )
    use_stabilized_llrd = (
        use_llrd
        and new_module_lr_raw is not None
        and float(new_module_lr_raw) > 0.0
    )
    new_module_param_ids: set[int] = set()

    if use_stabilized_llrd:
        variant = str(cfg.model.get("variant", "vit_b"))
        if variant not in _VIT_VARIANTS:
            raise SystemExit(f"Unknown ViT variant {variant!r} for LLRD.")
        depth = int(_VIT_VARIANTS[variant]["depth"])
        new_module_lr = float(new_module_lr_raw)
        param_groups = _build_llrd_stabilized_param_groups(
            model,
            base_lr=head_lr,
            new_module_lr=new_module_lr,
            weight_decay=weight_decay,
            layer_decay=float(layer_decay_raw),
            depth=depth,
            backbone_warmup_epochs=warmup_epochs,
            new_module_warmup_epochs=new_module_warmup_epochs,
        )
        new_module_param_ids = _new_module_param_ids(param_groups)
        print(
            f"[optim] stabilized LLRD: backbone lr={head_lr:g} (warmup {warmup_epochs} ep), "
            f"new_module lr={new_module_lr:g} (warmup {new_module_warmup_epochs} ep), "
            f"layer_decay={float(layer_decay_raw):g}, depth={depth}, "
            f"{len(param_groups)} param groups."
        )
    elif use_llrd:
        variant = str(cfg.model.get("variant", "vit_b"))
        if variant not in _VIT_VARIANTS:
            raise SystemExit(f"Unknown ViT variant {variant!r} for LLRD.")
        depth = int(_VIT_VARIANTS[variant]["depth"])
        param_groups = _build_llrd_param_groups(
            model,
            base_lr=head_lr,
            weight_decay=weight_decay,
            layer_decay=float(layer_decay_raw),
            depth=depth,
        )
        print(
            f"[optim] LLRD: layer_decay={float(layer_decay_raw):g}, depth={depth}, "
            f"{len(param_groups)} param groups."
        )
    elif use_dual_stream_groups:
        rgb_params, motion_params, rest_params = _split_dual_stream_backbone_params(model)
        if motion_lr_raw is not None:
            eff_motion_lr = float(motion_lr_raw)
            motion_lr_origin = "motion_lr (absolute)"
        else:
            eff_motion_lr = head_lr * float(motion_lr_ratio_raw)
            motion_lr_origin = f"motion_lr_ratio={float(motion_lr_ratio_raw):g}"
        param_groups: list[dict[str, Any]] = []
        if rgb_params:
            param_groups.append(
                {
                    "params": rgb_params,
                    "lr": head_lr,
                    "weight_decay": weight_decay,
                    "warmup_lr_max": head_lr,
                    "name": "rgb_backbone",
                }
            )
        if motion_params:
            param_groups.append(
                {
                    "params": motion_params,
                    "lr": eff_motion_lr,
                    "weight_decay": weight_decay,
                    "warmup_lr_max": eff_motion_lr,
                    "name": "motion_backbone",
                }
            )
        if rest_params:
            param_groups.append(
                {
                    "params": rest_params,
                    "lr": head_lr,
                    "weight_decay": weight_decay,
                    "warmup_lr_max": head_lr,
                    "name": "fuse_head",
                }
            )
        if not param_groups:
            raise RuntimeError(
                "Dual-stream param-group split produced 0 trainable groups; "
                "check that rgb_backbone / motion_backbone are not fully frozen."
            )
        print(
            f"[optim] dual-stream three-group LR ({motion_lr_origin}): "
            f"rgb lr={head_lr:g} (n={len(rgb_params)}), "
            f"motion lr={eff_motion_lr:g} (n={len(motion_params)}), "
            f"fuse_head lr={head_lr:g} (n={len(rest_params)})"
        )
    elif use_lora_group:
        param_groups = [
            {
                "params": head_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
                "warmup_lr_max": head_lr,
                "name": "head",
            },
            {
                "params": lora_params,
                "lr": eff_lora_lr,
                "weight_decay": weight_decay,
                "warmup_lr_max": eff_lora_lr,
                "name": "lora",
            },
        ]
        print(
            f"[optim] two param groups: head lr={head_lr:g} (n={len(head_params)}), "
            f"lora lr={eff_lora_lr:g} (n={len(lora_params)})"
        )
    else:
        trainable_params = head_params + lora_params
        param_groups = [
            {
                "params": trainable_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
                "warmup_lr_max": head_lr,
                "name": "trainable",
            },
        ]

    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for g in param_groups for p in g["params"])
    print(
        f"[optim] trainable {trainable_count:,} / total {total_params:,} params "
        f"({100.0 * trainable_count / max(1, total_params):.2f}%)"
    )
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=float(cfg.training.get("momentum", 0.9)),
            nesterov=bool(cfg.training.get("nesterov", False)),
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups)
    else:
        optimizer = torch.optim.Adam(param_groups)

    use_cosine = bool(cfg.training.get("scheduler_cosine", False))
    scheduler_name = str(cfg.training.get("scheduler", "cosine" if use_cosine else "none")).lower()
    cosine_start_epoch = (
        max(warmup_epochs, new_module_warmup_epochs)
        if use_stabilized_llrd
        else warmup_epochs
    )
    cosine_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    sgdr_T0 = int(cfg.training.get("sgdr_T0", 30))
    sgdr_save_snapshots = bool(cfg.training.get("sgdr_save_snapshots", False))
    if scheduler_name == "sgdr":
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, sgdr_T0),
            T_mult=1,
            eta_min=float(cfg.training.get("min_lr", 0.0)),
        )
        print(
            f"[sched] SGDR CosineAnnealingWarmRestarts T_0={sgdr_T0}, "
            f"eta_min={float(cfg.training.get('min_lr', 0.0)):g}"
        )
    elif use_cosine or scheduler_name == "cosine":
        cosine_tmax = max(1, int(cfg.training.epochs) - cosine_start_epoch)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_tmax,
            eta_min=float(cfg.training.get("min_lr", 0.0)),
        )

    label_smoothing = float(cfg.training.get("label_smoothing", 0.0))
    videomix_alpha = float(cfg.training.get("videomix_alpha", 0.0))
    videomix_prob = float(cfg.training.get("videomix_prob", 1.0))
    videomix_mode = str(cfg.training.get("videomix_mode", "cube_cutmix"))
    videomix_mixup_alpha = cfg.training.get("videomix_mixup_alpha")
    videomix_cutmix_alpha = cfg.training.get("videomix_cutmix_alpha")
    eff_videomix_mixup_alpha = (
        float(videomix_mixup_alpha) if videomix_mixup_alpha is not None else None
    )
    eff_videomix_cutmix_alpha = (
        float(videomix_cutmix_alpha) if videomix_cutmix_alpha is not None else None
    )
    videomix_switch_prob = float(cfg.training.get("videomix_switch_prob", 0.5))
    grad_accum_steps = max(1, int(cfg.training.get("grad_accum_steps", 1)))
    log_interval_steps = int(cfg.training.get("log_interval_steps", 0))
    early_stopping_enabled = bool(cfg.training.get("early_stopping_enabled", False))
    early_stopping_patience = int(cfg.training.get("early_stopping_patience", 10))
    early_stopping_min_delta = float(cfg.training.get("early_stopping_min_delta", 0.0))
    stop_on_mlp_activity_raw = cfg.training.get("stop_on_mlp_activity_ratio")
    stop_on_mlp_activity_ratio: float | None = (
        float(stop_on_mlp_activity_raw)
        if stop_on_mlp_activity_raw is not None
        else None
    )
    max_grad_norm_raw = cfg.training.get("max_grad_norm")
    max_grad_norm: float | None = (
        float(max_grad_norm_raw) if max_grad_norm_raw is not None else None
    )
    new_module_max_grad_norm_raw = cfg.training.get("new_module_max_grad_norm")
    new_module_max_grad_norm: float | None = (
        float(new_module_max_grad_norm_raw)
        if new_module_max_grad_norm_raw is not None
        else None
    )

    amp_enabled = bool(cfg.training.get("amp", False)) and device.type == "cuda"
    amp_dtype_str = str(cfg.training.get("amp_dtype", "float16")).lower()
    amp_dtype = (
        torch.bfloat16
        if amp_dtype_str in ("bf16", "bfloat16")
        else torch.float16
    )
    scaler: torch.amp.GradScaler | None = (
        torch.amp.GradScaler("cuda", enabled=True) if amp_enabled else None
    )
    if amp_enabled:
        print(f"[amp] mixed-precision ({amp_dtype} + GradScaler) enabled.")
    if grad_accum_steps > 1:
        print(f"[train] grad_accum_steps={grad_accum_steps} (effective batch scales up).")
    if max_grad_norm is not None and max_grad_norm > 0.0:
        print(f"[train] gradient clipping enabled (max_norm={max_grad_norm:g}).")
    if new_module_max_grad_norm is not None and new_module_max_grad_norm > 0.0:
        print(
            f"[train] new-module gradient clipping "
            f"(max_norm={new_module_max_grad_norm:g}, n_params={len(new_module_param_ids)})."
        )
    if stop_on_mlp_activity_ratio is not None:
        print(
            f"[train] will stop if head mlp_activity_ratio > "
            f"{stop_on_mlp_activity_ratio:g} at epoch end."
        )

    # EMA (Exponential Moving Average) of model weights, à la PyTorch's
    # ``AveragedModel(use_buffers=True)``. When enabled, every optimizer step
    # we update the EMA copy, and at each epoch boundary we evaluate *both*
    # the live model and the EMA model. The checkpoint stores whichever
    # achieved the best val_top1; the corresponding state_dict is recorded so
    # ``submit.py`` always loads the right weights without code changes.
    ema_enabled = bool(cfg.training.get("ema_enabled", False))
    ema_decay = float(cfg.training.get("ema_decay", 0.999))
    ema_model: torch.optim.swa_utils.AveragedModel | None = None
    if ema_enabled:

        def _ema_avg_fn(
            avg_param: torch.Tensor, model_param: torch.Tensor, _num_averaged: int
        ) -> torch.Tensor:
            return ema_decay * avg_param + (1.0 - ema_decay) * model_param

        ema_model = torch.optim.swa_utils.AveragedModel(
            model, avg_fn=_ema_avg_fn, use_buffers=True
        ).to(device)
        print(f"[ema] enabled (decay={ema_decay}); will eval both live and EMA each epoch.")

    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    best_top1 = -1.0
    best_path: Path | None = None
    epochs_without_improvement = 0
    start_epoch = 0

    save_last_enabled = bool(cfg.training.get("save_last_checkpoint", True))
    last_override = cfg.training.get("last_checkpoint_path")
    if last_override:
        last_checkpoint_path = Path(str(last_override)).resolve()
    else:
        last_checkpoint_path = checkpoint_path.with_name(
            checkpoint_path.stem + ".last" + checkpoint_path.suffix
        )

    def _save_last_checkpoint(
        epoch_done: int, latest_val_top1: float | None
    ) -> None:
        """Persist the live model + optimizer state at the end of ``epoch_done``."""
        if not save_last_enabled:
            return
        last_extra: dict[str, Any] = {
            "epoch": int(epoch_done),
            "val_top1": float(best_top1),
            "trained_class_indices": trained_class_indices,
            "checkpoint_kind": "last",
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if cosine_scheduler is not None:
            last_extra["scheduler_state_dict"] = cosine_scheduler.state_dict()
        if scaler is not None:
            last_extra["scaler_state_dict"] = scaler.state_dict()
        if latest_val_top1 is not None:
            last_extra["latest_val_top1"] = float(latest_val_top1)
        save_checkpoint(last_checkpoint_path, model, cfg, extra=last_extra)

    resume_from = cfg.training.get("resume_from")
    if resume_from:
        resume_path = Path(str(resume_from)).resolve()
        print(f"Resuming from checkpoint: {resume_path}")
        payload = load_checkpoint(resume_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        extra = payload.get("extra") or {}
        resume_reset_epoch = bool(cfg.training.get("resume_reset_epoch", False))
        prior_epoch = int(extra.get("epoch", 0))
        if resume_reset_epoch:
            start_epoch = 0
            best_top1 = -1.0
            best_path = None
            print(
                f"  Loaded weights from epoch {prior_epoch}; "
                f"resume_reset_epoch=true -> starting at epoch 0/"
                f"{int(cfg.training.epochs)} (best val reset for new split)."
            )
        else:
            start_epoch = prior_epoch
            if "val_top1" in extra:
                best_top1 = float(extra["val_top1"])
            best_path = resume_path
            print(
                f"  Resumed at epoch {start_epoch}/{int(cfg.training.epochs)}; "
                f"best val top1 so far = {best_top1:.4f}"
            )

        opt_restored = False
        opt_state = extra.get("optimizer_state_dict")
        if opt_state is not None:
            try:
                optimizer.load_state_dict(opt_state)
                opt_restored = True
                print("  Optimizer state restored from checkpoint.")
            except Exception as exc:
                print(f"  [warn] could not restore optimizer state: {exc}")

        sched_state = extra.get("scheduler_state_dict")
        scaler_state = extra.get("scaler_state_dict")
        if not opt_restored:
            sched_state = None
            scaler_state = None
            print(
                "  [resume] optimizer not restored; ignoring saved scheduler/scaler "
                "state (incompatible param groups or missing state). Cosine uses "
                "fast-forward from the fresh optimizer."
            )

        resume_apply_cfg_lr = bool(cfg.training.get("resume_apply_cfg_lr", False))
        if resume_apply_cfg_lr and opt_restored:
            for group in optimizer.param_groups:
                name = str(group.get("name", ""))
                if name == "motion_backbone":
                    peak = eff_motion_lr if use_dual_stream_groups else head_lr
                elif name == "lora":
                    peak = eff_lora_lr
                else:
                    peak = head_lr
                group["lr"] = peak
                group["warmup_lr_max"] = peak
            sched_state = None
            print(
                "  [resume] resume_apply_cfg_lr=true: applied cfg peak LRs; "
                "cosine scheduler will fast-forward from new bases."
            )

        if cosine_scheduler is not None and sched_state is not None:
            try:
                cosine_scheduler.load_state_dict(sched_state)
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"  Cosine scheduler state restored; resumed LR={current_lr:.6g}")
            except Exception as exc:
                print(f"  [warn] could not restore scheduler state: {exc}")
                sched_state = None

        if cosine_scheduler is not None and sched_state is None:
            ff_steps = max(0, start_epoch - cosine_start_epoch)
            for _ in range(ff_steps):
                cosine_scheduler.step()
            if ff_steps > 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  Cosine scheduler fast-forwarded {ff_steps} step(s); "
                    f"resumed LR={current_lr:.6g}"
                )

        if scaler is not None and scaler_state is not None:
            try:
                scaler.load_state_dict(scaler_state)
                print("  GradScaler state restored from checkpoint.")
            except Exception as exc:
                print(f"  [warn] could not restore GradScaler state: {exc}")

        lr_parts = [
            f"{g.get('name', f'group{i}')}={g['lr']:.6g}"
            for i, g in enumerate(optimizer.param_groups)
        ]
        sched_hint = (
            "cosine will step after each train epoch"
            if cosine_scheduler is not None
            else "fixed (scheduler_cosine=false)"
        )
        print(f"  LR after resume: {', '.join(lr_parts)} ({sched_hint}).")

    # Per-step stability logger for the HC/mHC ablation: grad norms (global +
    # per-block), HC mixing-matrix drift, mHC SK doubly-stochastic deviation.
    # Off by default; enabled by setting ``training.stability_log_path``.
    stability_log_path = cfg.training.get("stability_log_path")
    stability_logger = None
    if stability_log_path:
        from smth2smth.shared.engine.stability import StabilityLogger
        stability_logger = StabilityLogger(
            Path(str(stability_log_path)).resolve(),
            model=model,
            log_every=int(cfg.training.get("stability_log_every", 1)),
            include_per_block=bool(cfg.training.get("stability_log_per_block", True)),
            include_hc_drift=bool(cfg.training.get("stability_log_hc_drift", True)),
        )
        print(
            f"[stability] per-step logging -> {stability_log_path} "
            f"(every {int(cfg.training.get('stability_log_every', 1))} step(s))"
        )

    wandb_tracker = WandbTracker(cfg)
    steps_per_epoch = len(train_loader)

    try:
        for epoch in range(start_epoch, int(cfg.training.epochs)):
            epoch_step_offset = epoch * steps_per_epoch
            step_metrics_cb = build_step_metrics_callback(
                wandb_tracker, step_offset=epoch_step_offset
            )
            if scheduler_name != "sgdr":
                for group in optimizer.param_groups:
                    g_warm = int(group.get("warmup_epochs", warmup_epochs))
                    if g_warm > 0 and epoch < g_warm:
                        warm_scale = float(epoch + 1) / float(g_warm)
                        max_lr = float(group.get("warmup_lr_max", group["lr"]))
                        group["lr"] = max_lr * warm_scale
            if stability_logger is not None:
                stability_logger.set_epoch(epoch)
            train_stats: EpochStats = train_one_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                device,
                num_classes=int(cfg.num_classes),
                label_smoothing=label_smoothing,
                videomix_alpha=videomix_alpha,
                videomix_prob=videomix_prob,
                videomix_mode=videomix_mode,
                videomix_mixup_alpha=eff_videomix_mixup_alpha,
                videomix_cutmix_alpha=eff_videomix_cutmix_alpha,
                videomix_switch_prob=videomix_switch_prob,
                log_interval_steps=log_interval_steps,
                grad_accum_steps=grad_accum_steps,
                scaler=scaler,
                amp_dtype=amp_dtype,
                ema_model=ema_model,
                class_weights=class_weights,
                stability_logger=stability_logger,
                step_metrics_callback=step_metrics_cb,
                max_grad_norm=max_grad_norm,
                new_module_param_ids=new_module_param_ids or None,
                new_module_max_grad_norm=new_module_max_grad_norm,
            )
            eval_every_n_epochs = max(1, int(cfg.training.get("eval_every_n_epochs", 1)))
            eval_ema = bool(cfg.training.get("eval_ema", True))
            is_last_epoch = (epoch + 1) == int(cfg.training.epochs)
            should_eval = is_last_epoch or ((epoch + 1) % eval_every_n_epochs == 0)
            if not should_eval:
                et = int(cfg.training.epochs)
                pfx = _epoch_progress_stamp(epoch + 1, et)
                print(
                    f"{pfx}Epoch {epoch + 1}/{et} | "
                    f"train loss {train_stats.loss:.4f} top1 {train_stats.top1:.4f} | "
                    f"val skipped (eval_every_n_epochs={eval_every_n_epochs})"
                )
                if cosine_scheduler is not None and epoch >= cosine_start_epoch:
                    cosine_scheduler.step()
                if wandb_tracker.should_log_epoch(epoch + 1, int(cfg.training.epochs)):
                    log_epoch_summary(
                        wandb_tracker,
                        epoch_one_indexed=epoch + 1,
                        steps_per_epoch=steps_per_epoch,
                        train_stats=train_stats,
                        lr=float(optimizer.param_groups[0]["lr"]),
                        best_top1=best_top1,
                        optimizer=optimizer,
                        use_val_holdout=use_val_holdout,
                    )
                _save_last_checkpoint(epoch_done=epoch + 1, latest_val_top1=None)
                continue

            holdout_stats: EpochStats = evaluate_epoch(
                model,
                val_loader,
                loss_fn,
                device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            ema_holdout_stats: EpochStats | None = None
            if ema_model is not None and eval_ema:
                ema_holdout_stats = evaluate_epoch(
                    ema_model,
                    val_loader,
                    loss_fn,
                    device,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )
            honest_stats: EpochStats | None = None
            ema_honest_stats: EpochStats | None = None
            if val_honest_loader is not None:
                honest_stats = evaluate_epoch(
                    model,
                    val_honest_loader,
                    loss_fn,
                    device,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )
                if ema_model is not None and eval_ema:
                    ema_honest_stats = evaluate_epoch(
                        ema_model,
                        val_honest_loader,
                        loss_fn,
                        device,
                        amp_enabled=amp_enabled,
                        amp_dtype=amp_dtype,
                    )

            et = int(cfg.training.epochs)
            pfx = _epoch_progress_stamp(epoch + 1, et)
            if use_val_holdout:
                val_line = (
                    f"val holdout loss {holdout_stats.loss:.4f} top1 {holdout_stats.top1:.4f} "
                    f"top5 {holdout_stats.top5:.4f}"
                )
                if honest_stats is not None:
                    val_line += (
                        f" | val honest top1 {honest_stats.top1:.4f} "
                        f"top5 {honest_stats.top5:.4f}"
                    )
            else:
                val_line = (
                    f"val loss {holdout_stats.loss:.4f} top1 {holdout_stats.top1:.4f} "
                    f"top5 {holdout_stats.top5:.4f}"
                )
            if ema_holdout_stats is not None:
                ema_line = (
                    f"ema holdout top1 {ema_holdout_stats.top1:.4f} "
                    f"top5 {ema_holdout_stats.top5:.4f}"
                )
                if ema_honest_stats is not None:
                    ema_line += (
                        f" | ema honest top1 {ema_honest_stats.top1:.4f} "
                        f"top5 {ema_honest_stats.top5:.4f}"
                    )
                print(
                    f"{pfx}Epoch {epoch + 1}/{et} | "
                    f"train loss {train_stats.loss:.4f} top1 {train_stats.top1:.4f} | "
                    f"{val_line} | {ema_line}"
                )
            else:
                ema_tag = " | ema eval skipped" if (ema_model is not None and not eval_ema) else ""
                print(
                    f"{pfx}Epoch {epoch + 1}/{et} | "
                    f"train loss {train_stats.loss:.4f} top1 {train_stats.top1:.4f} | "
                    f"{val_line}{ema_tag}"
                )

            # Diverse-head failure-mode probes (Arch 1 MLP liveness / Arch 2
            # query collapse). Refreshed by the eval forward just run; no-op for
            # the mean-pool control and the temporal-only archs.
            head_diag = (
                model.head_diagnostics()
                if hasattr(model, "head_diagnostics")
                else {}
            )
            if head_diag:
                diag_str = ", ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in head_diag.items())
                print(f"  [head-diag] {diag_str}")
                mlp_ratio = head_diag.get("head/mlp_activity_ratio")
                if (
                    stop_on_mlp_activity_ratio is not None
                    and mlp_ratio is not None
                    and float(mlp_ratio) > stop_on_mlp_activity_ratio
                ):
                    print(
                        "Stopping: head mlp_activity_ratio "
                        f"{float(mlp_ratio):.4f} > {stop_on_mlp_activity_ratio:g} "
                        "(activation runaway guard)."
                    )
                    break

            # Pick the better of (live, EMA) for checkpointing. The chosen
            # state_dict is saved as ``model_state_dict`` so ``submit.py`` /
            # ``evaluate.py`` keep working without any "is this EMA?" branching.
            if ema_holdout_stats is not None and ema_holdout_stats.top1 > holdout_stats.top1:
                ckpt_stats, ckpt_module, ckpt_kind = ema_holdout_stats, ema_model, "ema"
            else:
                ckpt_stats, ckpt_module, ckpt_kind = holdout_stats, model, "live"

            if ckpt_stats.top1 > (best_top1 + early_stopping_min_delta):
                best_top1 = ckpt_stats.top1
                epochs_without_improvement = 0
                ckpt_extra: dict[str, Any] = {
                    "val_top1": ckpt_stats.top1,
                    "val_top5": ckpt_stats.top5,
                    "val_loss": ckpt_stats.loss,
                    "epoch": epoch + 1,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "trained_class_indices": trained_class_indices,
                    "checkpoint_kind": ckpt_kind,
                }
                if honest_stats is not None:
                    ckpt_extra["val_honest_top1"] = float(honest_stats.top1)
                    ckpt_extra["val_honest_top5"] = float(honest_stats.top5)
                if ema_honest_stats is not None:
                    ckpt_extra["val_ema_honest_top1"] = float(ema_honest_stats.top1)
                if cosine_scheduler is not None:
                    ckpt_extra["scheduler_state_dict"] = cosine_scheduler.state_dict()
                if scaler is not None:
                    ckpt_extra["scaler_state_dict"] = scaler.state_dict()
                module_to_save = (
                    ckpt_module.module
                    if isinstance(ckpt_module, torch.optim.swa_utils.AveragedModel)
                    else ckpt_module
                )
                best_path = save_checkpoint(
                    checkpoint_path,
                    module_to_save,
                    cfg,
                    extra=ckpt_extra,
                )
                print(
                    f"  Saved new best checkpoint ({ckpt_kind}): "
                    f"{best_path} (val top1={ckpt_stats.top1:.4f})"
                )
            else:
                epochs_without_improvement += 1
                if early_stopping_enabled:
                    print(
                        "  No val top1 improvement "
                        f"({epochs_without_improvement}/{early_stopping_patience}); "
                        f"best remains {best_top1:.4f}"
                    )
                    if epochs_without_improvement >= early_stopping_patience:
                        print(
                            "Early stopping triggered: "
                            f"no improvement > {early_stopping_min_delta:.6f} for "
                            f"{early_stopping_patience} consecutive epochs."
                        )
                        break

            if wandb_tracker.should_log_epoch(epoch + 1, int(cfg.training.epochs)):
                log_epoch_summary(
                    wandb_tracker,
                    epoch_one_indexed=epoch + 1,
                    steps_per_epoch=steps_per_epoch,
                    train_stats=train_stats,
                    val_holdout_stats=holdout_stats,
                    ema_holdout_stats=ema_holdout_stats,
                    val_honest_stats=honest_stats,
                    ema_honest_stats=ema_honest_stats,
                    lr=float(optimizer.param_groups[0]["lr"]),
                    best_top1=best_top1,
                    head_diag=head_diag if head_diag else None,
                    optimizer=optimizer,
                    use_val_holdout=use_val_holdout,
                )
            if cosine_scheduler is not None and (
                scheduler_name == "sgdr" or epoch >= cosine_start_epoch
            ):
                cosine_scheduler.step()

            if (
                sgdr_save_snapshots
                and scheduler_name == "sgdr"
                and (epoch + 1) % max(1, sgdr_T0) == 0
            ):
                snap_idx = (epoch + 1) // max(1, sgdr_T0)
                snap_path = checkpoint_path.with_name(
                    f"{checkpoint_path.stem}_snap{snap_idx}{checkpoint_path.suffix}"
                )
                snap_extra: dict[str, Any] = {
                    "val_top1": ckpt_stats.top1,
                    "val_top5": ckpt_stats.top5,
                    "val_loss": ckpt_stats.loss,
                    "epoch": epoch + 1,
                    "trained_class_indices": trained_class_indices,
                    "checkpoint_kind": "sgdr_snapshot",
                    "sgdr_cycle": snap_idx,
                }
                save_checkpoint(snap_path, model, cfg, extra=snap_extra)
                print(f"  [sgdr] saved cycle-{snap_idx} snapshot -> {snap_path}")

            _save_last_checkpoint(epoch_done=epoch + 1, latest_val_top1=ckpt_stats.top1)
    except torch.cuda.OutOfMemoryError as exc:
        print(f"[cuda] OOM during training: {exc}. Releasing memory and aborting this job.")
        del model, optimizer, train_loader, val_loader
        _free_cuda_memory(reason="post-OOM")
        raise
    finally:
        if stability_logger is not None:
            stability_logger.close()
        wandb_tracker.finish()
        _free_cuda_memory(reason="run-end")

    if best_path is None:
        print("Training finished without producing a checkpoint.")
    else:
        val_label = "holdout" if use_val_holdout else "honest"
        print(
            f"Done. Best val {val_label} top1: {best_top1:.4f}. Checkpoint: {best_path}"
        )
    return best_path


@hydra.main(version_base=None, config_path=CONFIGS_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra CLI entrypoint."""
    run(cfg)


if __name__ == "__main__":
    main()
