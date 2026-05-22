# Track B (Open World) — Experiment Design, 2026-05-20

Context inputs:
- Deep Research output and reconciliation in [track_b_next_steps.md](track_b_next_steps.md).
- Current Track B backbone wrapper: [src/smth2smth/track_b/vjepa2.py](src/smth2smth/track_b/vjepa2.py).
- Existing PEFT recipe baseline: [configs/experiment/track_b_vjepa2_peft.yaml](configs/experiment/track_b_vjepa2_peft.yaml).
- Recent overfitting evidence: [track_b_vitl_fulltrain_lowlr_20260516_1740.log](logs/track_b_vitl_fulltrain_lowlr_20260516_1740.log) — train top-1 ≈ 0.94 while val saturates at 0.8919.
- Local class subset is 32 (not 33): `ls data/train/` shows folder 027 missing; mirror pair `018_Pulling_..._left_to_right` / `019_Pulling_..._right_to_left` is present, no Pushing pair.

Bottlenecks identified:
1. Encoder mismatch — current best uses the SSL-only `vjepa2-vitl-fpc64-256` base; Meta's `vjepa2-vitl-fpc16-256-ssv2` is already supervised on the full 174-class SSv2 and its head can be sliced to our 32 indices for a much stronger init.
2. TTA mis-shape — 2×3×2 (flip) TTA on a direction-sensitive dataset regressed −9.6 pp; the official V-JEPA recipe is 2 seg × 3 crops, no flip.
3. Probe capacity — [AttentiveProbe](src/smth2smth/track_b/vjepa2.py#L132-L198) is a single-block, multi-query MHA; Meta's reported 73.7 % uses a 4-block, 16-head attentive classifier.
4. Train/val gap is large (≈5 pp at epoch 10) — no EMA, no spatial VideoMix; the LoRA-only knobs (`r=8`, attention-only targets) sit below the published sweet spot.
5. Temporal distribution shift — dataloader feeds 4 real frames duplicated 4× into a 16-slot tensor (verify in [video_dataset.py](src/smth2smth/shared/data/video_dataset.py)); the SSv2-FT head was trained on real 16-frame sequences. This is the project's novelty angle and an unmeasured ceiling.

---

### Experiment 1: SSv2-FT checkpoint swap + head-slice + LoRA at 16 frames

Hypothesis: Replacing `facebook/vjepa2-vitl-fpc64-256` (SSL-only) with `facebook/vjepa2-vitl-fpc16-256-ssv2` (Meta's own supervised SSv2 finetune, 73.7 % top-1 on the full 174-class val) and slicing its 174-class head down to our 32 indices preserves the trained class prototypes for every class we care about. The 141 dropped classes can no longer steal probability mass, so even before training the zero-shot baseline should jump well above the current 45 %. A small LoRA on top adapts the encoder to our 4-frame-duplicated-to-16 input distribution. This is the single highest-EV lever in the open-world setting because it imports supervised SSv2 knowledge that we are explicitly allowed to use.

Implementation Details:
- [src/smth2smth/track_b/vjepa2.py](src/smth2smth/track_b/vjepa2.py): switch `DEFAULT_HF_REPO` plumbing path so the builder loads `VJEPA2ForVideoClassification` (not `AutoModel`) when `head_init_from_pretrained=true`; copy the 32 selected rows of `classifier.weight/bias` into our [AttentiveProbe.classifier](src/smth2smth/track_b/vjepa2.py#L178) (replace the `nn.init.normal_(..., std=0.01)`).
- New helper `scripts/build_local_to_ssv2_idx.py` that reads `data/train/*/`, strips the `NNN_` prefix, looks up each name in `VJEPA2ForVideoClassification.from_pretrained("facebook/vjepa2-vitl-fpc16-256-ssv2").config.label2id`, and saves `local_to_ssv2_idx.pt` of shape `(32,)`. Fail loudly on any unmatched folder (precedent: truncated class 015 token-align fallback in the existing zero-shot path [src/smth2smth/track_b/zero_shot.py](src/smth2smth/track_b/zero_shot.py)).
- New config `configs/experiment/track_b_vjepa2_ssv2ft_lora16f.yaml`: `hf_repo=facebook/vjepa2-vitl-fpc16-256-ssv2`, `num_frames=16`, LoRA `r=16`, `alpha=32`, `target_modules=[q_proj,k_proj,v_proj,o_proj]`, head lr 5e-4, LoRA lr 1e-4, AdamW wd 0.05, cosine, 15 epochs, bf16, no random hflip.
- Add a calibration entry to [src/smth2smth/track_b/zero_shot.py](src/smth2smth/track_b/zero_shot.py) so the sliced head is evaluated before training (gate per §"Calibration run" in [track_b_next_steps.md](track_b_next_steps.md)).

Estimated Effort: Medium

Priority: 1

---

### Experiment 2: Replace flip-augmented TTA with the official V-JEPA 2×3-no-flip protocol

Hypothesis: The current 12-view (2 seg × 3 crops × 2 flips) TTA averages logits across flipped views on a dataset where direction is a label feature, structurally pulling direction-sensitive classes toward the wrong logit; this explains the −9.6 pp regression we observed. The official Meta recipe is `num_segments=2, num_views_per_segment=3, no flip` (codified in `configs/eval/vitg-384/ssv2.yaml` and the checkpoint filename `ssv2-vitl-16x2x3.pt`). Matching it should recover most of that loss for +2 to +3 pp on top of any Cycle 1 winner. The existing `(18,19)` Pulling-L↔R remap is correct (verified by `ls data/train/`) and must be kept.

Implementation Details:
- [src/smth2smth/pipelines/](src/smth2smth/pipelines/) submission/eval pipeline: drop the flip view, switch to softmax-averaging (not logit-averaging), keep the local-index `(18,19)` remap.
- Update [src/smth2smth/track_b/zero_shot.py](src/smth2smth/track_b/zero_shot.py) inference loop to emit a 2-segments × 3-spatial-crops view stack at the chosen `image_size`.
- One submission config delta in `configs/experiment/track_b_vjepa2_ssv2ft_lora16f.yaml` (`tta.flip=false`, `tta.num_views_per_segment=3`).

Estimated Effort: Low

Priority: 2

---

### Experiment 3: 4-block / 16-head attentive classifier (Meta architecture) + last-2-block token concat

Hypothesis: The current head is a single-block multi-query MHA on the last encoder layer. Meta's published 73.7 % on SSv2 ViT-L/256 uses a 4-block, 16-head attentive classifier on the last 2 encoder blocks concatenated (DINOv2 / PE-Core convention). Concatenating tokens from the last 2 blocks roughly doubles the feature channel dim and recovers fine-grained motion cues that a single-block probe drops. This is the largest probe-side change available and stacks orthogonally with Experiment 1.

Implementation Details:
- [src/smth2smth/track_b/vjepa2.py](src/smth2smth/track_b/vjepa2.py): add `AttentiveClassifier4Block` mirroring `AttentiveClassifier(embed_dim=1024, num_heads=16, depth=4, num_classes=32)` from `facebookresearch/vjepa2/notebooks/vjepa2_demo.ipynb`. Wire a new `head_type="vjepa_4block"` branch in `_build_head` (around [src/smth2smth/track_b/vjepa2.py#L237](src/smth2smth/track_b/vjepa2.py#L237)).
- Modify `VJEPA2Probe._encode` to request hidden states (`output_hidden_states=True`) and concatenate the last K=2 along the channel dim; expose `head_last_k_blocks: int = 1` in the config.
- New config `configs/experiment/track_b_vjepa2_ssv2ft_4block.yaml` inheriting from Experiment 1 with `head_type=vjepa_4block`, `head_last_k_blocks=2`, sweep lr ∈ {3e-4, 1e-4}, wd ∈ {0.01, 0.1}, warmup 0.

Estimated Effort: High

Priority: 3

---

### Experiment 4: EMA over probe + LoRA params, with EMA-checkpoint evaluation

Hypothesis: The training logs show overfitting (train top-1 0.94 at epoch 10 vs val 0.89; same pattern across LoRA runs). EMA over the small set of trainable parameters (probe + LoRA adapters ≈ 12 MB) is one of the highest-confidence "+0.5 to +1.5 pp generalization wins" in the published literature and is essentially free on a 3090. Because our trainable subset is tiny relative to the backbone, EMA cost is negligible.

Implementation Details:
- Add `torch.optim.swa_utils.AveragedModel` with a custom `avg_fn` for decay 0.9998 around the optimizer step in the training loop (locate via `grep -rn "loss.backward" src/smth2smth/pipelines/`).
- Only EMA `model.head.*` and any `lora_` parameter names; skip frozen backbone params explicitly to keep the EMA shadow small.
- At validation and submission, swap to the EMA-shadow weights for forward.
- Make EMA opt-in via `training.ema_enabled` and `training.ema_decay` in a new config that extends Experiment 1.

Estimated Effort: Low

Priority: 4

---

### Experiment 5: LoRA upgrade — r=16, α=32, MLP targets, plus a DoRA arm

Hypothesis: Track B's current PEFT recipe ([configs/experiment/track_b_vjepa2_peft.yaml](configs/experiment/track_b_vjepa2_peft.yaml#L17)) uses `r=8` and attention-only targets (`q,k,v,proj`). The literature consensus is `r=16–32, α=2r` plus MLP targets (`mlp.fc1, mlp.fc2`) for new-task adaptation — torchtune reports ~+4 pp on truthfulqa from this exact change. DoRA (Liu ICML 2024) further decomposes magnitude from direction and is a drop-in via PEFT 0.10+'s `use_dora=True`; CLIP-DoRA reports a +0.28 % avg in the vision regime — small but free given the configuration cost. Running r=8 vs r=16, LoRA vs DoRA in parallel on two machines (per §D) gives ablation data the report needs.

Implementation Details:
- [src/smth2smth/track_b/vjepa2.py](src/smth2smth/track_b/vjepa2.py#L113): extend `LoraConfig(...)` with `use_dora=cfg.model.dora_enabled`. Plumb `dora_enabled` through `build_vjepa2` ([src/smth2smth/track_b/vjepa2.py#L436](src/smth2smth/track_b/vjepa2.py#L436)).
- Update `_DEFAULT_LORA_TARGETS` to include MLP names once verified against the HF V-JEPA2 module graph (`grep -rn "mlp\.fc" $(python -c 'import transformers; print(transformers.__path__[0])')/models/vjepa2/`). Expose them in config to avoid hardcoding.
- Two new configs extending Experiment 1: `..._lora16_mlp.yaml` and `..._dora16_mlp.yaml`. Same seed, same data split, only the PEFT config differs — guarantees a clean ablation row.

Estimated Effort: Low

Priority: 5

---

### Experiment 2b: TTA protocol (baked into E1 config)

The 2×3 no-flip TTA is already set in `track_b_vjepa2_ssv2ft_lora16f.yaml` (`tta_flip: false`, `test.num_segment: 2`, `test.num_crop: 3`). There is no separate E2 training run — the TTA fix is applied at submit time and inherited by all configs that extend E1. The `(18, 19)` Pulling-L↔R remap is computed automatically in `submit.py` from class folder names.

---

### Experiment 6: 4-frame-regime full-SSv2 augmentation + frozen-probe baseline (novelty angle)

Hypothesis: The local dataset feeds 4 real frames duplicated 4× into a 16-slot tensor (the Kaggle test set is in the same distribution). Meta's SSv2-FT head has never seen this zero-motion-delta token structure. Two outputs from one experimental track: (a) a *frozen-probe* baseline ("no LoRA, head-only") quantifies the ceiling under temporal distribution shift — directly the novelty story for the professor (4-frame → 16-frame transfer is unaddressed in the V-JEPA 2 paper); (b) pulling the full SSv2 train videos for our 32 classes, re-downsampling to 4 frames at the professor's exact stride, then duplicating to 16, gives ~10× more training rows in the *correct* distribution — a labeled-data win the rules permit in the open-world track.

Implementation Details:
- New script `scripts/download_ssv2_subset_4frame.py`: pulls SSv2 source clips for the 32 class names, applies the project's frame-sampler (first 60 % of frames → subsample to 4 → duplicate 4×), writes to `data/train_extra/` (do not overwrite `data/train/`). Verify byte-equality on one overlapping clip vs the professor's distribution before bulk download.
- `data/manifest.py` (or [src/smth2smth/shared/data/video_dataset.py](src/smth2smth/shared/data/video_dataset.py)): teach the dataset registry about `train_extra/` so it is concatenated with `train/` only when `data.use_extra=true`.
- Two configs extending Experiment 1: `..._frozen_probe.yaml` (LoRA disabled, head-only) and `..._extra_train.yaml` (LoRA on, extra data on). Run frozen-probe first; it is the single-variable ablation against Experiment 1 (LoRA on/off) that the report needs.
- Open verification first (per §"Open verification items"): write a 30-line script that loads one clip from the existing dataloader, asserts `tensor[:, 0]==tensor[:, 1]==tensor[:, 2]==tensor[:, 3]` etc. to confirm the 4-dup-to-16 layout before committing to the re-downsampling pipeline.

Verified frame layout (empirical, `scripts/verify_4frame_dup_layout.py` on `data/train/000_Closing_something/video_10061`):
- 4 frames on disk; sampler indices `[0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3]`; duplication groups `[[0,1,2],[3,4,5,6,7],[8,9,10,11,12],[13,14,15]]` (sizes 3,5,5,3).
- NOT even 4× duplication — the linspace sampler produces uneven repeats when 4 does not divide 16 evenly.
- The download script stores 4 real frames per extra clip (matching the on-disk layout). The dataloader's `pick_frame_indices` handles 4→16 expansion identically for train and extra clips.

Estimated Effort: High

Priority: 6

---

## How to Launch (SSH Agent Reference)

### Environment

```bash
# Working directory: /Data/thomas.turkieh/smth2smth
# Python: 3.12.9 via uv (pyproject.toml specifies exact version)
# CUDA: 12.8  GPU: RTX 3090 (24 GB VRAM)

# Install / sync dependencies (run once per machine)
uv sync

# All commands below assume cwd = repo root
# PYTHONPATH is handled automatically by Hydra (pythonpath = ["src"] in pyproject.toml)
```

### Launch commands

All training runs use the same Hydra entrypoint. The `experiment=<name>` override selects the config from `configs/experiment/<name>.yaml`.

```bash
# E1 — SSv2-FT checkpoint swap + head-slice + LoRA r=16 (HIGHEST PRIORITY — run first)
uv run python -m smth2smth.pipelines.train experiment=track_b_vjepa2_ssv2ft_lora16f

# E3 — 4-block attentive head + last-2-block concat (SSL base, NOT SSv2-FT)
uv run python -m smth2smth.pipelines.train experiment=track_b_vjepa2_4block_lastk2

# E4 — EMA decay=0.9998 on the SSL-base LoRA recipe
uv run python -m smth2smth.pipelines.train experiment=track_b_vjepa2_vitl_ema9998

# E5a — LoRA r=16 + MLP targets (fc1/fc2), extending E1
uv run python -m smth2smth.pipelines.train experiment=track_b_vjepa2_lora16_mlp

# E5b — DoRA r=16 + MLP targets, extending E5a (run on second machine in parallel with E5a)
uv run python -m smth2smth.pipelines.train experiment=track_b_vjepa2_dora16_mlp

# E6a — Frozen-probe baseline: SSv2-FT encoder, NO LoRA, head-only (run before E6b)
uv run python -m smth2smth.pipelines.train experiment=track_b_vjepa2_ssv2ft_frozen_probe

# E6b — SSv2-FT + LoRA + extra SSv2 training data (requires E6b prerequisite below)
uv run python -m smth2smth.pipelines.train experiment=track_b_vjepa2_ssv2ft_extra_train
```

### Submission / inference

```bash
uv run python -m smth2smth.pipelines.submit experiment=track_b_vjepa2_ssv2ft_lora16f
# Swap experiment= to use a different checkpoint. TTA: 2 segments × 3 crops, no flip.
```

### E6b prerequisite: populate data/train_extra/

E6b requires `data/train_extra/` to exist with SSv2 source clips re-downsampled to 4 frames. Needs: (1) SSv2 source `.webm` files and (2) SSv2 `train.json` labels (registration at 20bn.com required).

```bash
# Step 0: verify the local frame layout (should print "4 unique frames" with sizes 3,5,5,3)
uv run python scripts/verify_4frame_dup_layout.py --expect-unique 4

# Step 1: dry-run the plan (no decode, no writes — confirms class matching)
uv run python scripts/download_ssv2_subset_4frame.py \
    --ssv2-labels-json /path/to/ssv2/train.json \
    --local-classes-dir data/train

# Step 2: extract (needs PyAV — already in pyproject.toml as av>=17.0.1)
uv run python scripts/download_ssv2_subset_4frame.py \
    --ssv2-labels-json /path/to/ssv2/train.json \
    --ssv2-videos-dir /path/to/ssv2/20bn-something-something-v2 \
    --local-classes-dir data/train \
    --out-dir data/train_extra \
    --limit-per-class 200 \
    --no-dry-run

# Step 3: launch E6b
uv run python -m smth2smth.pipelines.train experiment=track_b_vjepa2_ssv2ft_extra_train
```

### Checkpoint locations

| Experiment | Config file | Checkpoint saved to |
|---|---|---|
| E1 | `track_b_vjepa2_ssv2ft_lora16f` | `checkpoints/track_b/ssv2ft_lora16f.pt` |
| E3 | `track_b_vjepa2_4block_lastk2` | `checkpoints/track_b/vitl_4block_lastk2.pt` |
| E4 | `track_b_vjepa2_vitl_ema9998` | `checkpoints/track_b/vitl_ema9998.pt` |
| E5a | `track_b_vjepa2_lora16_mlp` | `checkpoints/track_b/ssv2ft_lora16_mlp.pt` |
| E5b | `track_b_vjepa2_dora16_mlp` | `checkpoints/track_b/ssv2ft_dora16_mlp.pt` |
| E6a | `track_b_vjepa2_ssv2ft_frozen_probe` | `checkpoints/track_b/ssv2ft_frozen_probe.pt` |
| E6b | `track_b_vjepa2_ssv2ft_extra_train` | `checkpoints/track_b/ssv2ft_extra_train.pt` |

### Config inheritance map

```
track_b_vjepa2_vitl_30e_warmup  (old SSL base — E3, E4 still extend this)
├── track_b_vjepa2_4block_lastk2           (E3)
└── track_b_vjepa2_vitl_ema9998            (E4)

track_b_vjepa2_ssv2ft_lora16f              (E1 — SSv2-FT + LoRA r=16)
├── track_b_vjepa2_lora16_mlp              (E5a — adds MLP LoRA targets)
│   └── track_b_vjepa2_dora16_mlp         (E5b — DoRA instead of LoRA)
├── track_b_vjepa2_ssv2ft_frozen_probe     (E6a — no LoRA)
└── track_b_vjepa2_ssv2ft_extra_train      (E6b — extra data on)
```

### Priority order for a single GPU

1. **E1** — `track_b_vjepa2_ssv2ft_lora16f` (highest EV: SSv2-FT init + head-slice)
2. **E6a** — `track_b_vjepa2_ssv2ft_frozen_probe` (ablation baseline for the novelty story; depends on E1 finishing to compare)
3. **E5a** — `track_b_vjepa2_lora16_mlp` (MLP LoRA targets delta over E1)
4. **E5b** — `track_b_vjepa2_dora16_mlp` (DoRA arm — run on a second machine in parallel with E5a if available)
5. **E6b** — `track_b_vjepa2_ssv2ft_extra_train` (requires SSv2 source data; run last or concurrently on a second machine)
6. **E3** / **E4** — lower priority; still on the old SSL base, not yet ported to the SSv2-FT encoder

### Notes for the SSH agent

- **HuggingFace model download**: `facebook/vjepa2-vitl-fpc16-256-ssv2` (~1.2 GB) is downloaded automatically on first run by `transformers`. Ensure HF_HOME or cache dir has enough space and network access. If offline, pre-cache with `huggingface-cli download facebook/vjepa2-vitl-fpc16-256-ssv2`.
- **AMP / bf16**: All E1-derived configs use `amp: true`. The 3090 supports bf16; if running on an older card, add `training.amp=false` as a Hydra override.
- **Resume**: Add `training.resume_from=checkpoints/track_b/<name>.pt` to resume an interrupted run.
- **Logs**: Hydra outputs go to `outputs/<date>/<time>/`. The training loop also appends to `logs/` with a timestamped filename.
- **Data paths**: Default data root is `data/` relative to repo root. Override with `dataset.root=/abs/path` if data lives elsewhere.
- **Class 027**: This local class folder is absent (only 32 of 33 classes exist). The head-slice builder handles this silently by randomly initializing that row. All configs are aware.
- **The 4-frame distribution shift**: Local clips have 4 real frames on disk; `pick_frame_indices(4, 16)` expands to 16 via linspace rounding (groups of sizes 3,5,5,3 — not even 4×). The SSv2-FT head was trained on real 16-frame sequences. This distribution mismatch is the paper's novelty claim; E6a/E6b address it experimentally.
