# V63: question-conditioned whole-answer image assignment

Registered 5 November 2026 before production implementation or new model
evaluation. The V62 primary and auxiliary reports remain immutable. Its
mixture improves bitmap fit but not useful question-dependent answers.
V29 already tested a two-by-two assignment critic unsuccessfully, and V37–38
tested pooled semantic targets without a working image writer. V63 does not
claim the assignment identity is new. It tests that supervision directly in
the proper likelihood of a fully trainable, free-image language generator.

## Data and image-only boundary

Use the unchanged V61/V62 Noto-26 binary image cache and CMRC source split:
3,211 training questions, 1,321 reused development questions. Group training
questions by exact context hash. Enumerate sorted pairs with disjoint retained
reference strings and disjoint canonical reference-image sequences. Only
training data and reference pixels enter admission. Record the complete pair
manifest, rejection counts and hashes before production. Initial text-only
count is 3,793 pairs covering 3,016 questions; physical aliases may reduce it.
Do not select pairs from model scores. Sample contexts uniformly, then valid
pairs uniformly within context, using the same RNG/order in both arms.

For two prompt images X0, X1 and their actual answer-image suffixes Y0, Y1,
construct all four streams by joining each prompt to each answer. Suffixes
include the unchanged padding and two white termination blocks. No text,
Unicode/character IDs, OCR, answer offset in the paragraph, candidate bank or
external model call reaches the actor. Full teacher-forced streams are causal;
the next physical image is supervised only after its preceding image state.
Inference generates its own pixels with the unchanged V62 runtime.

## Objective and controlled comparison

Let eij be mean answer/termination-pixel negative log likelihood of Yj given
Xi under the normalized eight-expert bitmap density. Define

    D = e01 + e10 - e00 - e11
    Lpositive = (e00 + e11) / 2
    Lassignment = 0.01 * softplus(-D / 0.01)
    L = Lpositive + weight * Lassignment.

The aligned arm uses weight 1; the control uses weight 0. Both compute the
same four likelihood graphs and receive identical sampled training pairs.
Pure row or column energy offsets cancel from D; an answer marginal alone
cannot solve the assignment. This algebra is not a guarantee that the learned
state captures meaning or that likelihood ranking transfers to free writing.
Different prompt lengths and visual cues remain possible shortcuts, so actual
generation, question blanking and held-out paragraph pairs remain required.

Both arms start from V62 K=8 final EMA, SHA-256
`9e22ebccbd769cc44e7de026ca89e433ae934ddeaf808af01020552f24d1b1b9`.
Architecture and participating parameter count are identical: 489,998,208.
Every backbone and rendering parameter trains. No extra semantic head,
teacher feature or changed font is introduced in this comparison.

Run 2,000 updates per arm, microbatch one pair, accumulation four pairs
(eight positive examples per update), AdamW 2e-5, betas (.9,.95), decay .01,
100-update warmup, cosine decay to .1, norm clip one, EMA .99. Seeds are
20266320 for model/training and 20266321 for pair sampling. FP32 model,
optimizer and EMA; BF16 forward/backward; deterministic eager/math attention
and nonreentrant activation checkpointing. Use final EMA only, without
development checkpoint or seed selection. Run arms sequentially on physical
GPU 0, at most 20 GiB allocated memory and eight hours per training arm.
Never use GPU 1 or another project's model service.

## Real preflight before production

Verify physical prefix/suffix assembly in all four streams; scalar loss and
row/column cancellation; finite nonzero assignment-only gradients through
input, first/last contextual layers, gate and physical projection; and zero
future-answer effect on earlier predictions. Measure one full optimizer/EMA
update on the longest admitted training pair with all four graphs retained.
If it does not fit, stop and revise the implementation plan before production;
do not change the data or objective to hide the failure.

For each arm compare four real accumulated updates with two plus resumed two,
requiring bit-exact model, EMA, optimizer and all RNG states. Smoke states are
implementation evidence only. Source/data/initialization hashes must match on
resume. Keep complete pair order, and preserve any uncommitted journal tail
when resuming a checkpoint after an interrupted save. A zero process exit
does not imply completion without the final 2,000-update report.

## Locked audit and decision

Finish both arms before examining development generations. Reuse the exact
256 primary IDs, all 393 shared-paragraph pairs and all 1,321 likelihood
examples. Report stopped exact raster answers, multiset raster F1, aligned
foreground F1, empty/cap/repeat rates, and both-correct pairs, with normal and
blank-question controls in both arms. Also report four-way reference-answer
energies and assignment sign on all 393 pairs as a separate ranking diagnostic;
it cannot substitute for autonomous generation.

Use article-grouped paired bootstrap, 2,000 draws, seed 20266329. A bounded
reading pass retains exact answers >=10%, raster F1 >=25%, positive lower
normal-minus-blank F1 interval, >=2% both-correct pairs, cap rate <20%, and all
boundary, finite/resource, resume/causality and offline replay checks. The
assignment intervention advances only if the aligned arm passes those reading
gates and its raster-F1 gain over matched control has a positive lower interval.
Keep all raw outputs and the first eight fixed primary rows without selection.
Do not introduce a new OCR/vision-judge metric into this locked decision.

Verify native-PNG generation and learned standalone bundles in fresh offline
processes with empty source/model caches, comparing output tensors, PNG bytes
and events. Report external foundation pretraining separately from local
adaptation, plus loading, training, audit and peak allocated memory. Reused
development and one seed preclude a final generalization or parity claim.
Even a bounded pass leaves broad instruction following, book reading,
historical scripts and general-language quality unresolved.
