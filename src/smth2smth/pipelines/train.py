"""Training pipeline.

Run from the repo root::

    PYTHONPATH=src uv run python -m smth2smth.pipelines.train experiment=baseline_pretrained track=a

Tests can call :func:`run` directly with a hand-built ``DictConfig``; only the
:func:`main` wrapper depends on Hydra.
"""

from __future__ import annotations

import gc
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
from smth2smth.shared.utils import (
    compute_class_weights,
    compute_sample_weights,
    set_seed,
    split_train_val,
)

CONFIGS_DIR = str(Path(__file__).resolve().parents[3] / "configs")


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


def _split_trainable_params_head_vs_lora(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Partition trainable parameters into probe/head vs PEFT LoRA adapters.

    HuggingFace PEFT names adapter weights with ``lora_A`` / ``lora_B`` in the
    parameter name. Everything else trainable is treated as the head (or
    non-LoRA trainables).

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
        if "lora_A" in name or "lora_B" in name:
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
    if use_official_val:
        # Validate on the official held-out folder. The internal 80/20 split is
        # bypassed: training uses *all* of ``train_dir``, validation uses
        # *all* of ``val_dir``.
        val_dir_for_val = Path(cfg.dataset.val_dir).resolve()
        val_samples = collect_video_samples(val_dir_for_val)
        if max_samples is not None:
            val_samples = val_samples[: int(max_samples)]
        train_samples = all_samples
        print(
            f"[data] use_official_val=true: train={len(train_samples)} "
            f"(from {train_dir}), val={len(val_samples)} (from {val_dir_for_val})"
        )
    else:
        train_samples, val_samples = split_train_val(
            all_samples,
            val_ratio=float(cfg.dataset.val_ratio),
            seed=int(cfg.dataset.seed),
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
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

    model = build_model(cfg).to(device)

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

    if use_dual_stream_groups:
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
    warmup_epochs = int(cfg.training.get("warmup_epochs", 0))
    cosine_scheduler = None
    if use_cosine:
        cosine_tmax = max(1, int(cfg.training.epochs) - warmup_epochs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_tmax,
            eta_min=float(cfg.training.get("min_lr", 0.0)),
        )

    label_smoothing = float(cfg.training.get("label_smoothing", 0.0))
    videomix_alpha = float(cfg.training.get("videomix_alpha", 0.0))
    videomix_prob = float(cfg.training.get("videomix_prob", 1.0))
    videomix_mode = str(cfg.training.get("videomix_mode", "cube_cutmix"))
    log_interval_steps = int(cfg.training.get("log_interval_steps", 0))
    early_stopping_enabled = bool(cfg.training.get("early_stopping_enabled", False))
    early_stopping_patience = int(cfg.training.get("early_stopping_patience", 10))
    early_stopping_min_delta = float(cfg.training.get("early_stopping_min_delta", 0.0))

    amp_enabled = bool(cfg.training.get("amp", False)) and device.type == "cuda"
    scaler: torch.amp.GradScaler | None = (
        torch.amp.GradScaler("cuda", enabled=True) if amp_enabled else None
    )
    if amp_enabled:
        print("[amp] mixed-precision (fp16 + GradScaler) enabled.")

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
        start_epoch = int(extra.get("epoch", 0))
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
            ff_steps = max(0, start_epoch - warmup_epochs)
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

    try:
        for epoch in range(start_epoch, int(cfg.training.epochs)):
            if warmup_epochs > 0 and epoch < warmup_epochs:
                warm_scale = float(epoch + 1) / float(warmup_epochs)
                for group in optimizer.param_groups:
                    max_lr = float(group.get("warmup_lr_max", group["lr"]))
                    group["lr"] = max_lr * warm_scale
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
                log_interval_steps=log_interval_steps,
                scaler=scaler,
                ema_model=ema_model,
                class_weights=class_weights,
            )
            eval_every_n_epochs = max(1, int(cfg.training.get("eval_every_n_epochs", 1)))
            eval_ema = bool(cfg.training.get("eval_ema", True))
            is_last_epoch = (epoch + 1) == int(cfg.training.epochs)
            should_eval = is_last_epoch or ((epoch + 1) % eval_every_n_epochs == 0)
            if not should_eval:
                print(
                    f"Epoch {epoch + 1}/{cfg.training.epochs} | "
                    f"train loss {train_stats.loss:.4f} top1 {train_stats.top1:.4f} | "
                    f"val skipped (eval_every_n_epochs={eval_every_n_epochs})"
                )
                if cosine_scheduler is not None and epoch >= warmup_epochs:
                    cosine_scheduler.step()
                _save_last_checkpoint(epoch_done=epoch + 1, latest_val_top1=None)
                continue

            val_stats: EpochStats = evaluate_epoch(
                model, val_loader, loss_fn, device, amp_enabled=amp_enabled
            )
            ema_stats: EpochStats | None = None
            if ema_model is not None and eval_ema:
                ema_stats = evaluate_epoch(
                    ema_model, val_loader, loss_fn, device, amp_enabled=amp_enabled
                )
            if ema_stats is not None:
                print(
                    f"Epoch {epoch + 1}/{cfg.training.epochs} | "
                    f"train loss {train_stats.loss:.4f} top1 {train_stats.top1:.4f} | "
                    f"val loss {val_stats.loss:.4f} top1 {val_stats.top1:.4f} "
                    f"top5 {val_stats.top5:.4f} | "
                    f"ema val top1 {ema_stats.top1:.4f} top5 {ema_stats.top5:.4f}"
                )
            else:
                ema_tag = " | ema eval skipped" if (ema_model is not None and not eval_ema) else ""
                print(
                    f"Epoch {epoch + 1}/{cfg.training.epochs} | "
                    f"train loss {train_stats.loss:.4f} top1 {train_stats.top1:.4f} | "
                    f"val loss {val_stats.loss:.4f} top1 {val_stats.top1:.4f} "
                    f"top5 {val_stats.top5:.4f}{ema_tag}"
                )

            # Pick the better of (live, EMA) for checkpointing. The chosen
            # state_dict is saved as ``model_state_dict`` so ``submit.py`` /
            # ``evaluate.py`` keep working without any "is this EMA?" branching.
            if ema_stats is not None and ema_stats.top1 > val_stats.top1:
                ckpt_stats, ckpt_module, ckpt_kind = ema_stats, ema_model, "ema"
            else:
                ckpt_stats, ckpt_module, ckpt_kind = val_stats, model, "live"

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
            if cosine_scheduler is not None and epoch >= warmup_epochs:
                cosine_scheduler.step()
            _save_last_checkpoint(epoch_done=epoch + 1, latest_val_top1=ckpt_stats.top1)
    except torch.cuda.OutOfMemoryError as exc:
        print(f"[cuda] OOM during training: {exc}. Releasing memory and aborting this job.")
        del model, optimizer, train_loader, val_loader
        _free_cuda_memory(reason="post-OOM")
        raise
    finally:
        _free_cuda_memory(reason="run-end")

    if best_path is None:
        print("Training finished without producing a checkpoint.")
    else:
        print(f"Done. Best val top1: {best_top1:.4f}. Checkpoint: {best_path}")
    return best_path


@hydra.main(version_base=None, config_path=CONFIGS_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra CLI entrypoint."""
    run(cfg)


if __name__ == "__main__":
    main()
