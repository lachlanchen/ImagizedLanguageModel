# Real conditional-writing preflight for occupancy hierarchy

The new component's [structure audit](occupancy_density_structure_result_20261118.md)
passes on every saved-support cell. This protocol admits only a training-only
head comparison after its implementation, sources and registration are committed
and pushed. The old reader, modulation, density and language failures remain
terminal. No full model fit or autonomous evaluation is admitted here.

Reuse the immutable actual conditional cache from the completed writer-layer
preflight. It contains the original final actor's states before every next
span, target images and latents, for the same three complete 513-, 846- and
1,042-glyph training records. Do not rerun its reader/body/codec or load a teacher
head. Verify its hash against that original result. There are 512, five and
35 next-span glyphs; no current target conditions its own state. Repeat each
span's prefix state for its physical glyph positions and supply the index and
span length. This component factorizes glyphs within a supplied span; this
limitation cannot be relabelled as full span-language modeling.

Compare flat and tree occupancy laws with identical 2,261,074-parameter tensor
shapes and the same initial draw, seed 202611190. Both have the same learned
empty/nonempty root. Both generate topology and every physical bit without a
glyph inventory or reference topology. The root choices initially agree.
Root errors, nonempty garbage and damaged glyphs remain scored.

Before fitting, accumulate one actual whole-record image-likelihood gradient
for each case in each arm. Require finite nonzero contextual-leaf, root and
mask-output gradients. No optimizer step follows these probes. Keep all inherited
actors and cached inputs unchanged. Then run exactly **96 AdamW updates per
arm**, cycling through the three whole sources 32 times: learning rate 2e-4,
weight decay .01, global clip one. Accumulate each complete record's mean
total-image NLL divided by 1,024 over microbatches of eight images before one
step. No source truncation, vocabulary objective, margin or reward is added.
Use FP32 parameters and head execution, with TF32 disabled. Source states retain
their original precision in the immutable cache and are converted to FP32 at
this new head's input.

Before and after fitting, generate all 552 images per arm in batches of sixteen,
greedily from the actual causal image states and physical span indices/lengths.
Every root and child mask is predicted. Persist every binary output, generated
tree trace, per-event probability and conditional NLL. Check cached versus
parallel probability replay at every active generated event within 2e-4 absolute
log-probability error. A mismatch stops the run without another update. Recount
source-pixel F1, exact masks, true spaces, false blanks and false ink. Do not
repair any output with a target, font, OCR or nearest image.

Also record correct-state and within-record one-position-rotated-state image
NLL, holding target prefixes and physical indices/lengths fixed. Count unchanged
state rows in this intervention and retain all positions. This is a conditional
dependence diagnostic, not a semantic judge or a training loss. It supplies no
new labels or rewards.

Physical support requires: final tree pooled F1 at least .75; every complete
case F1 at least .60; tree pooled F1 at least .05 above flat; zero false blanks
and zero false ink on all 552 training-support positions; and a pooled
rotated-minus-correct NLL advantage at least .5 nat per glyph. All source,
causality, gradient and replay checks must pass. These new component-support
conditions do not replace or relax any earlier gate or full-language requirement.
No single easy QA may replace the complete support.

Reset every trial tensor exactly to its own common initial state and discard
optimizer state. Retain final hashes and every actual output; install no trained
successor. A failed preflight stops this intervention at this budget. A pass
permits designing a separately registered complete-source comparison, with the
unchanged reader qualification, autonomous emitted-image feedback, paired QA,
erasures, stopping, books, retention and untouched confirmation. It is not a
language result or automatic permission to fit.

Use one serial GPU0 owner under the existing mutex; two CPU threads; entry
48 GiB available host RAM, 20 GiB free GPU and 48 GiB disk at low PSI; runtime
host reserve 24 GiB, maximum head allocation eight GiB. Monitored deadline
1,800 seconds, pressure-wait allowance 300 seconds; external deadline 1,860
seconds plus thirty-second termination grace. Preserve every failed execution
and use no automatic retry. These head costs exclude the inherited
1,465,381,142-parameter actor, its prior training and the reused neural cache's
preparation. No compact-foundation or matched-quality efficiency claim follows.
