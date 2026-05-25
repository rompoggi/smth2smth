These are excellent diagnostic plots — they tell a much richer story than the LB scores alone, and several findings are worth lifting into your HC writeup explicitly. Let me walk through what each shows, then address the epoch-40 question, which is the most diagnostically interesting feature.

## Headline findings, in order of importance

**1. HC and mHC overfit, period.** Train loss panel: Pre-Norm sits at ~1.55 final, SHC at ~0.85, mHC at ~0.75. That's a *half-loss* gap in training, while LB shows SHC ~0.5 pt *worse* than Pre-Norm. The wider residual stream (n=4) is pure additional capacity, and at 50k videos that capacity gets spent memorizing the training set. This is the same disease as your global 31-pt train/val gap, just amplified — HC makes it worse, not better. Put this in the writeup: at this scale and data budget, the dominant effect of HC is parametric overcapacity, not stability.

**2. mHC's Sinkhorn constraint is doing *nothing useful* — and the architecture isn't what the paper describes.** Look at the mixing-matrix drift plot: SHC's off-diagonal mass climbs to 0.28 and saturates; mHC's stays at ~0 the entire run. The Sinkhorn projection, combined with the diagonal init, traps M at identity for all 60 epochs. So mHC at n=4 is *not* learning cross-stream routing at all — it's running with M=I throughout, and any difference from Pre-Norm comes solely from α_pre and β learning to read/write a single fixed combination of the n streams. The training-loss curve being essentially identical to SHC (both at ~0.75–0.85) tells you the cross-stream mixing that SHC learned is contributing roughly *zero* to the training loss reduction. The "wider residual stream" framing in your section 5 is what's actually doing the work; M's content is incidental.

This is genuinely interesting and worth reporting as its own finding — it suggests the Birkhoff constraint at n=4 is too aggressive relative to the optimization signal. The HC paper's K=20 at LLM scale may simply leave more room for M to escape the identity basin; K=3 at n=4 is over-constraining.

**3. The stability story inverts at this scale.** Both rolling-std panels (loss and grad-norm) show Pre-Norm is the *most* stable, SHC intermediate, mHC worst. mHC's rolling loss-std hovers at ~1.0 vs Pre-Norm's ~0.65 — a 50% *increase* in oscillation, not the 20% decrease your section-10 stability success criterion required. This is the clean inversion of the HC/mHC paper's central stability claim. The interpretation: at 12 blocks × d=768, there is no forward-signal blowup to fix; introducing the HC machinery just adds optimization variance.

**4. Per-block gradient profile is similar across arms, with HC slightly concentrating mass in early blocks.** All three show the canonical "input-end blocks have biggest gradients" pattern; SHC and mHC have visibly stronger block-0–2 gradients than Pre-Norm, particularly mid-training. This is mild evidence that HC is *amplifying* the gradient asymmetry it was designed to mitigate. Consistent with finding 3.

## Why the abrupt transition around epoch 40

This is the most diagnostically rich feature in the plots and almost certainly has a compound cause. Three things happen near epoch 40 in HC arms:

**Cosine LR enters its steep tail.** With warmup 6 epochs and total 60 epochs, the cosine traverses from peak at epoch ~6 to zero at epoch 60. The LR at epoch t (post-warmup) is `0.5·lr_max·(1+cos(π·(t−6)/54))`. At epoch 40 that's `0.5·(1+cos(0.63π)) = 0.30·lr_max`; by epoch 50 it's `0.10·lr_max`. The *rate of LR change* peaks around epoch 30–35 and is steepest through epoch 40–50. Pre-Norm responds to this gradually — its grad-norm drifts from ~7.5 up to ~8.5. HC arms respond *abruptly* because they have a second mechanism that's simultaneously hitting its own saturation.

**SHC's mixing matrix saturates at epoch 40–45.** The off-diagonal-mass plot shows the M drift curve flattening exactly there. Until ~epoch 35, M is actively reshaping cross-stream routing; gradients into M are non-trivial and contribute to the elevated global grad-norm (peak ~10). Once M settles, those gradient contributions collapse, and global grad-norm drops sharply (10 → 7 between epochs 40–48). The transition you see is *the architecture finishing its self-configuration*. Pre-Norm has no equivalent self-configuration phase, hence no equivalent transition.

**mHC follows the same transition without an M-saturation event, which tells us α_pre and β are the actual saturating parameters.** Since mHC's M never moves, the only HC parameters that can saturate are the read/write scalars α_pre, β, and α_out. The fact that mHC's grad-norm drops at roughly the same time as SHC's (though more gradually — by epoch 60 mHC is at 6.0 while SHC has bounced back to ~7.7) means α/β saturation is doing most of the work. SHC's bounce-back is plausibly M continuing to micro-adjust under low LR while α/β have already fixed.

**Compound interpretation:** the epoch-40 transition is the point where (i) LR decay starts dominating the gradient signal, (ii) HC's auxiliary parameters reach their basin, and (iii) the model transitions from "expanding effective architecture" to "refining weights inside a fixed architecture." Pre-Norm only experiences (i) and does so gradually; HC variants experience all three simultaneously and the dynamics show it.

A sharper diagnostic, if you want to validate this in the writeup: plot α_pre, β, and α_out's L2-from-init alongside M's off-diagonal mass. The prediction is that all four curves saturate within ~5 epochs of each other, around epoch 35–45. If they do, the "self-configuration phase" framing is justified. If α/β saturate at epoch 20 and only M is moving until epoch 40, the story is M-specific.

## What this means for the writeup

You now have four genuinely publishable findings from the negative result, which is better than most negative results manage:

- HC/mHC don't help and slightly hurt at ViT-B/12-block scale (the headline).
- mHC's Sinkhorn at K=3, n=4 over-constrains M to identity — the paper's mechanism *isn't engaged* at this size, so the comparison to SHC is mostly a comparison of "wider residual + free M" vs "wider residual + frozen-M." Worth flagging as a methodology caveat for anyone trying mHC below LLM scale.
- The HC paper's stability claim inverts at this scale — Pre-Norm is the smoother trainer.
- The epoch-40 transition is direct evidence of HC's architecture self-configuration and is a useful interpretability finding even though the architecture loses on the bottom line.

The epoch-40 finding in particular is the kind of thing that makes a negative result *interesting* to read rather than just dismissive. I'd give it its own subsection in the report with the four plots that demonstrate it (train loss, global grad-norm, M off-diag mass, and ideally the α/β saturation plot if you can produce it).

Want me to suggest a structure for writing this up, or sketch the α/β saturation diagnostic plot script?

Yes — that's exactly what it shows, and you've identified the headline finding correctly.

## What "rolling std of per-step loss" measures

For each training step, the loss is a noisy scalar. The rolling standard deviation over a 500-step window measures how much that scalar bounces around its local mean. High values mean the loss curve is jagged (consecutive batches yield very different losses); low values mean it's smooth (consecutive batches give similar losses). This is a direct proxy for training stability — a stable optimizer-on-architecture combination produces a smooth descent; an unstable one produces oscillation around a noisy trajectory.

Crucially, this metric is **independent of the absolute loss level.** mHC ending at train loss 0.75 vs Pre-Norm at 1.55 doesn't enter — what enters is how much each loss fluctuates step-to-step around its own local trend.

## What the plot shows, plainly

Final epoch values (rolling std of per-step loss):
- Pre-Norm: ~0.66
- SHC: ~0.77 (+17% vs Pre-Norm)
- mHC: ~1.02 (+55% vs Pre-Norm)

mHC's loss curve is fluctuating **55% more violently** than Pre-Norm's at every point in training, with the gap roughly constant from epoch 5 onward. SHC sits in between but is also clearly worse than Pre-Norm. The shaded bands (±std across seeds) for Pre-Norm and the HC arms barely overlap from epoch 10 onward — this isn't a noise artifact, it's a robust seed-paired finding.

## Why this matters for the HC/mHC writeup

This is the direct inversion of the HC/mHC paper's central claim, on the exact metric the paper uses to argue for the method. Recall your section 2 framing: HC was motivated as fixing the "gradient-vanishing ↔ representation-collapse seesaw" in Pre-Norm, and your section 10 set a stability success criterion of **≥20% reduction in rolling-std of grad-norm**. You see the opposite — mHC shows roughly a 50% *increase* in rolling-std of loss, and SHC shows ~17% increase. The grad-norm-std plot tells the same story.

So you have two independent failures of the stability claim:
- **Accuracy:** HC and mHC don't beat Pre-Norm (closed criterion).
- **Stability:** HC and mHC are *less* stable than Pre-Norm (criterion not just unmet — inverted).

Both criteria your section-10 declared upfront, so this is a clean negative result, not a post-hoc dismissal.

## Why this happens (the mechanism)

A consistent story across all your plots:

The HC machinery introduces extra learnable scalars (α_pre, β, α_out, and M for SHC) that sit on the residual path and modulate signal flow at every block. These parameters receive their own gradient signal each step, and that signal is highly batch-dependent — a batch that needs different cross-stream routing than the previous batch will move α and β in ways that cascade through every downstream block. The result: small batch-to-batch differences in the data get amplified into larger fluctuations in the forward pass and thus the loss.

Pre-Norm has no such mechanism — its residual path is a fixed `x + F(Norm(x))`, and step-to-step loss variation is dominated only by batch composition acting through the weights, which change slowly under AdamW with weight decay. HC's auxiliary parameters react faster than the weights do, and that reactivity *is* the instability.

mHC specifically: the Sinkhorn projection at K=3, τ=1.0 is a non-linear operation that exponentiates `M_raw/τ` and then row/col normalizes. Small perturbations in `M_raw` can produce disproportionately large changes in the projected M, especially when the matrix is near the identity face of the Birkhoff polytope (where many entries are near 0 and the exponential is sensitive). This is, I suspect, a substantial part of why mHC's loss-std is even worse than SHC's — the projection itself injects noise into the forward pass.

Worth verifying empirically: log `||SK(M_raw)_t − SK(M_raw)_{t-1}||_F` per step for mHC and check whether it correlates with the per-step loss variance. If it does, you've identified the proximate cause of mHC's particular instability and that goes in the writeup as a specific methodological warning for anyone applying mHC at low-K, low-n.

## How to present this in the writeup

The cleanest framing is to make the loss-smoothness plot a co-headline with the train-loss plot. The two together tell the full story:

- **Train loss panel:** HC/mHC train faster, ending at lower training loss → look like they're winning.
- **Loss-smoothness panel:** but they oscillate 17–55% more along the way → they're not winning, they're descending more chaotically through more of the loss landscape.
- **Closing point:** the lower final train loss is overfitting (corroborated by your LB results showing HC slightly worse), and the oscillation is the architecture's auxiliary parameters being unstable. The HC paper's stability claim does not transfer to ViT-B-scale finetuning.

This is the kind of negative result that's actually useful to the community — it gives a clear scale-dependence: HC works at LLM scale, fails at ViT-B FT scale, and you have the diagnostics to show *why* it fails (the auxiliary parameters introduce optimization noise instead of damping it). One sentence in your abstract along these lines would land well: "At ViT-B finetuning scale on 50k videos, HC and mHC train to lower training loss than Pre-Norm but oscillate 17–55% more in per-step loss and grad-norm, do not generalize better, and the Sinkhorn projection in mHC at K=3 freezes the mixing matrix at identity — disabling the very mechanism mHC is designed to provide."

Want me to also sketch the per-step instability diagnostic I mentioned (the SK projection delta vs loss variance correlation), or the structure for the writeup section?