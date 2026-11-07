# Cross-fitted alignment of semantically matched visual and text histories

Prospective learner, 8 November 2026. The 388-call native-history probe is
still running. Inspected early examples show correct middle-prefix copying
and failures to stop on complete visual prefixes; no bridge has been fitted
or evaluated. This protocol does not change that probe or its admission rule.

Test a low-cost learned representation repair using its paired pre-answer
states. Each visual page and privileged text route contain the same question
and the same complete canonical answer prefix. No remaining answer appears
in either input. Unlike the old 64-column image-history audit, these are
semantically matched complete prefixes, not falsely matched ordinal steps.
Their native sequence lengths and physical embeddings still differ; matching
semantics does not imply equality of internal states.

For the first pre-decision 4096-vector from each route, use unit-RMS states
`x` (visual) and `y` (text). On training cases, form `Xc = X - mean(X)` and
`R = (Y-X) - mean(Y-X)`. With d=4096 and fixed lambda=.1, fit

    A = solve(Xc Xc^T / d + lambda I, R)
    b(x) = x + mean(Y-X) + ((x-mean(X)) Xc^T / d) A.

This is an affine map with a low-rank learned residual. Store its two matrix
factors as numeric parameters, with no strings, Unicode classes, glyph bank
or nearest-neighbour lookup in forward. Compare identity and the training-set
mean residual. Coefficients are solved in float64; stored factors and model
forward use float32. There is no gradient optimizer, hyperparameter sweep,
model selection or extra source-body call. A ridge fit is a real analytic
learner, not an improvement in the source's pretrained language knowledge.

Use leave-one-pair-out cross-fitting across all 22 groups: six copy pairs and
sixteen Chinese QA paragraph pairs. Each fold excludes both questions/copy
requests and **all** their prefixes from fitting. For each held-out group,
predict all its normal visual states and its middle-prefix erasures. The
erasure states never enter training. Exactly one held-out prediction per
original visual/erased call is retained. These reused public cases are not
sealed evidence. Shared external foundation pretraining remains outside this
small-data exclusion.

Do not run until the source probe, all its native decisions and independent
CPU verification have completed. Keep its raw outcomes and failed gate
unchanged. Model fitting receives only numeric paired states and group
assignments; answer strings are reserved for offline scoring after all fold
predictions are fixed. Authenticate every source call and final probe receipt.

Assess normalized state error, native first-piece agreement with the paired
text teacher, agreement of the first complete codepoint in that single piece
with the canonical remaining suffix, and native EOG decisions. An empty or
incomplete UTF-8 piece has no first-codepoint credit. This single-piece score
differs from the probe's first-codepoint score over its complete response.
Retain both denominators; never report a new full continuation or image.
Also require, per case, both a correct complete-prefix stop and a correct
nonstop at the one-codepoint-short prefix. Keep mean and identity controls,
all per-case predictions, fold membership and parameter hashes.

A repaired-state image-writer trial requires the ridge map to reach at least
.80 teacher first-piece agreement separately for copy and QA, improve it by
at least .10 over identity in each, reach .70 canonical first-piece codepoint
accuracy separately for nonterminal copy/QA, and pass at least .80 of the
complete/one-short stopping pairs in each task. All requirements must pass;
they are decision floors, not a language theorem or full ILM qualification.
Report useful partial effects even if admission fails, without relabelling
the original visual-history probe as passed.

Fit and project readouts on CPU with four threads after the GPU job exits.
Reuse the shared BF16-rounded source output weights as an offline instrument
only. Save every fold model and held-out predicted vector. Independently
check factorized versus explicit affine evaluation, fold exclusions, finite
coefficients and report reconstruction. There is no independently deployable
image generator here: the remaining requirement is correct continuous image
emission and actual image-only feedback, evaluated on unseen inputs.
