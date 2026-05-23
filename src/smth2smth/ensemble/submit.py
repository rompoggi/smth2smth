"""Ensemble test submission from cached or freshly computed per-member logits."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

from smth2smth.ensemble.combiners import sanitize_logits
from smth2smth.ensemble.inference import TtaMode, collect_logits_and_probs_for_videos, collect_logits_for_videos
from smth2smth.ensemble.optimize import combine_logits as combine_logits_scalar
from smth2smth.pipelines.submit import _build_untrained_mask, _resolve_test_videos
from smth2smth.shared.io.submission import write_submission_csv

_SEED_RE = re.compile(r"s(\d+)")


def load_experiment_weights(cache_dir: Path, exp_id: str) -> tuple[list[int], np.ndarray, TtaMode]:
    """Load member seeds, weight vector, and test TTA mode for ``exp_id``."""
    results_path = cache_dir / "ensemble_results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"Run optimize first: missing {results_path}")
    data = json.loads(results_path.read_text(encoding="utf-8"))
    block = data["weights"][exp_id]
    members = [int(s) for s in block["members"]]
    weights = np.array(block["weights"], dtype=np.float64)
    test_mode = TtaMode(block["test_mode"])
    return members, weights, test_mode


def _parse_v2_exp(exp_id: str) -> tuple[list[int], str, str, str]:
    """Return (members, fit_branch, eval_branch, combiner)."""
    if exp_id.startswith("S0-") or exp_id.startswith("S1-"):
        m = _SEED_RE.search(exp_id)
        if not m:
            raise ValueError(f"Cannot parse seed from {exp_id}")
        seed = int(m.group(1))
        branch = "basic" if exp_id.startswith("S0-") else "tta"
        return [seed], branch, branch, "single"
    parts = exp_id.split("-", 1)
    if len(parts) != 2:
        raise ValueError(f"Bad v2 exp id: {exp_id}")
    proto, comb = parts[0], parts[1]
    fit_b, eval_b = PROTOCOLS_V2[proto]
    return [42, 43, 44], fit_b, eval_b, comb


PROTOCOLS_V2 = {
    "T": ("tta", "tta"),
    "BT": ("basic", "tta"),
    "B": ("basic", "basic"),
}


def _test_logit_path(cache_dir: Path, seed: int, branch: str, *, tta_tag: str) -> Path:
    if branch == "basic":
        return cache_dir / f"logits_test_s{seed}_none.npy"
    return cache_dir / f"logits_test_s{seed}_{tta_tag}.npy"


def _test_prob_path(cache_dir: Path, seed: int, *, tta_tag: str) -> Path:
    return cache_dir / f"probs_test_s{seed}_{tta_tag}_probs.npy"


def _load_v2_row(cache_dir: Path, exp_id: str) -> dict:
    results_path = cache_dir / "ensemble_results.json"
    if not results_path.is_file():
        results_path = cache_dir / "ensemble_results_v2.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))
    for row in data["holdout_metrics"]:
        if row["exp"] == exp_id:
            return row
    raise KeyError(f"exp {exp_id} not in ensemble_results_v2.json")


def _combine_test_logits(
    exp_id: str,
    row: dict,
    logit_arrays: list[np.ndarray],
    prob_arrays: list[np.ndarray] | None,
) -> np.ndarray:
    comb = row["combiner"]
    weights = row.get("weights")
    if comb == "single":
        return logit_arrays[0]
    if comb in ("mean", "ws"):
        w = np.array(weights if comb == "ws" else [1 / len(logit_arrays)] * len(logit_arrays))
        if isinstance(weights, dict):
            members = [42, 43, 44]
            w = np.array([float(weights[str(s)]) for s in members], dtype=np.float64)
        return combine_logits_scalar(logit_arrays, w)
    if comb == "softmax":
        from smth2smth.ensemble.combiners import logits_to_probs

        if prob_arrays is None:
            prob_arrays = [logits_to_probs(a) for a in logit_arrays]
        return np.log(np.clip(np.mean(prob_arrays, axis=0), 1e-12, 1.0))
    if comb == "vote":
        preds = np.stack([np.argmax(a, axis=1) for a in logit_arrays], axis=1)
        n_cls = logit_arrays[0].shape[1]
        out = np.zeros((preds.shape[0], n_cls), dtype=np.float32)
        for i in range(preds.shape[0]):
            out[i, int(np.argmax(np.bincount(preds[i], minlength=n_cls)))] = 1.0
        return out.astype(np.float64)
    if comb == "cws":
        w = np.array(weights, dtype=np.float64)
        stacked = np.stack(logit_arrays, axis=0)
        return np.einsum("mnc,mc->nc", stacked, w)
    if comb == "lsg":
        coef = np.array(weights, dtype=np.float64)
        x = np.concatenate(logit_arrays, axis=1)
        return (x @ coef.T).astype(np.float64) if coef.ndim == 2 else x @ coef
    raise ValueError(f"Unsupported combiner for submit: {comb}")


def run_ensemble_submit_v2(
    *,
    exp_id: str,
    cache_dir: Path,
    checkpoints_dir: Path,
    output_csv: Path,
    test_root: Path,
    train_dir: Path,
    test_manifest: Path | None = None,
    batch_size: int = 8,
    num_workers: int = 4,
    force: bool = False,
    tta_tag: str = "champion",
) -> Path:
    """Write Kaggle CSV for a grid row (``T-ws``, ``BT-mean``, singles, …)."""
    row = _load_v2_row(cache_dir, exp_id)
    members, _fit_b, eval_b, _comb = _parse_v2_exp(exp_id)

    video_names, video_dirs = _resolve_test_videos(test_root, test_manifest)
    samples = [(p, 0) for p in video_dirs]

    logit_arrays: list[np.ndarray] = []
    prob_arrays: list[np.ndarray] | None = [] if eval_b == "tta" else None
    num_classes: int | None = None
    trained_indices: list[int] | None = None

    for seed in members:
        ckpt = checkpoints_dir / f"s{seed}_ft_val90_holdout.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

        if eval_b == "basic":
            cache_path = _test_logit_path(cache_dir, seed, "basic", tta_tag=tta_tag)
            if cache_path.is_file() and not force:
                logits = np.load(cache_path)
            else:
                print(f"[submit] test basic seed={seed} ...", flush=True)
                logits_t = collect_logits_for_videos(
                    ckpt,
                    samples,
                    data_root=test_root,
                    train_dir=train_dir,
                    tta_mode=TtaMode.NONE,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
                logits = logits_t.numpy()
                np.save(cache_path, logits)
            logit_arrays.append(sanitize_logits(logits))
        else:
            log_path = _test_logit_path(cache_dir, seed, "tta", tta_tag=tta_tag)
            prob_path = _test_prob_path(cache_dir, seed, tta_tag=tta_tag)
            if log_path.is_file() and not force:
                logits = np.load(log_path)
                probs = (
                    np.load(prob_path)
                    if prob_path.is_file()
                    else None
                )
            else:
                print(f"[submit] test {tta_tag} seed={seed} ...", flush=True)
                logits_t = collect_logits_for_videos(
                    ckpt,
                    samples,
                    data_root=test_root,
                    train_dir=train_dir,
                    tta_mode=TtaMode.CHAMPION,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
                logits = logits_t.numpy()
                np.save(log_path, logits)
                probs = None
            logit_arrays.append(sanitize_logits(logits))
            assert prob_arrays is not None
            prob_arrays.append(probs)

        if num_classes is None:
            num_classes = int(logit_arrays[-1].shape[1])
            from smth2smth.shared.io.checkpoints import load_checkpoint

            ck = load_checkpoint(ckpt, map_location="cpu")
            trained_indices = (ck.get("extra") or {}).get("trained_class_indices")

    combined = _combine_test_logits(exp_id, row, logit_arrays, prob_arrays)
    device = torch.device("cpu")
    mask = _build_untrained_mask(trained_indices, num_classes or combined.shape[1], device)
    logits_t = torch.from_numpy(combined.astype(np.float32))
    if mask is not None:
        logits_t = logits_t + mask
    predictions = logits_t.argmax(dim=1).cpu().tolist()

    output_csv = output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_path = write_submission_csv(output_csv, video_names, predictions)
    meta = {
        "exp_id": exp_id,
        "members": members,
        "eval_branch": eval_b,
        "combiner": row["combiner"],
        "weights": row.get("weights"),
        "n_videos": len(video_names),
        "output_csv": str(csv_path),
    }
    meta_path = cache_dir / f"submit_v2_{exp_id.replace('-', '_')}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[submit] v2 {exp_id} -> {csv_path} ({len(predictions)} rows)")
    return csv_path


def run_ensemble_submit(
    *,
    exp_id: str,
    cache_dir: Path,
    checkpoints_dir: Path,
    output_csv: Path,
    test_root: Path,
    train_dir: Path,
    test_manifest: Path | None = None,
    batch_size: int = 8,
    num_workers: int = 4,
    force_logits: bool = False,
) -> Path:
    """Write a Kaggle-style CSV for a learned-weight ensemble experiment.

    Args:
        exp_id: Key in ``ensemble_results.json`` (e.g. ``P3``).
        cache_dir: Directory with ``ensemble_results.json`` and optional cached logits.
        checkpoints_dir: Contains ``s{seed}_ft_val90_holdout.pt`` per member.
        output_csv: Destination CSV path.
        test_root: Test frame root (``data/test``).
        train_dir: Training root (flip-pair remap).
        test_manifest: Optional manifest listing test video order.
        batch_size: Inference batch size per member.
        num_workers: DataLoader workers.
        force_logits: Recompute ``logits_test_*`` even if cached.

    Returns:
        Resolved path to the written CSV.
    """
    members, weights, test_mode = load_experiment_weights(cache_dir, exp_id)
    video_names, video_dirs = _resolve_test_videos(test_root, test_manifest)
    samples = [(p, 0) for p in video_dirs]

    logit_arrays: list[np.ndarray] = []
    num_classes: int | None = None
    trained_indices: list[int] | None = None

    for seed in members:
        ckpt = checkpoints_dir / f"s{seed}_ft_val90_holdout.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        cache_path = cache_dir / f"logits_test_s{seed}_{test_mode.value}.npy"
        if cache_path.is_file() and not force_logits:
            logits = np.load(cache_path)
            print(f"[submit] loaded {cache_path.name} shape={logits.shape}")
        else:
            print(f"[submit] computing test logits seed={seed} mode={test_mode.value} ...", flush=True)
            logits_t = collect_logits_for_videos(
                ckpt,
                samples,
                data_root=test_root,
                train_dir=train_dir,
                tta_mode=test_mode,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            logits = logits_t.numpy()
            np.save(cache_path, logits)
            print(f"[submit] wrote {cache_path.name}")

        if logits.shape[0] != len(samples):
            raise RuntimeError(
                f"logits rows {logits.shape[0]} != videos {len(samples)} for seed {seed}"
            )
        logit_arrays.append(logits)
        if num_classes is None:
            num_classes = int(logits.shape[1])
            from smth2smth.shared.io.checkpoints import load_checkpoint

            ck = load_checkpoint(ckpt, map_location="cpu")
            trained_indices = (ck.get("extra") or {}).get("trained_class_indices")

    assert num_classes is not None
    combined = combine_logits_scalar(logit_arrays, weights)
    device = torch.device("cpu")
    mask = _build_untrained_mask(trained_indices, num_classes, device)
    logits_t = torch.from_numpy(combined.astype(np.float32))
    if mask is not None:
        logits_t = logits_t + mask
    predictions = logits_t.argmax(dim=1).cpu().tolist()

    output_csv = output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_path = write_submission_csv(output_csv, video_names, predictions)
    meta = {
        "exp_id": exp_id,
        "members": members,
        "weights": weights.tolist(),
        "test_mode": test_mode.value,
        "n_videos": len(video_names),
        "output_csv": str(csv_path),
    }
    meta_path = cache_dir / f"submit_{exp_id}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[submit] ensemble {exp_id} -> {csv_path} ({len(predictions)} rows)")
    print(f"[submit] weights seed order {members}: {weights}")
    return csv_path
