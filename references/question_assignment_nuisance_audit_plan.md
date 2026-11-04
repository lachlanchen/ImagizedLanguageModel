# Training-only audit of coarse visual assignment cues

Specified 5 November 2026 before running this diagnostic. V63 production is
already running; its sources, data, sampling, objective and decision gates
remain frozen. This audit reads no development predictions or sealed tests
and makes no model calls. It is not an estimate of Bayesian susceptibility.

The recent [physical-data susceptibility study](https://arxiv.org/html/2611.02632v1)
motivates asking which properties of training examples drive a capability.
Before introducing a costly posterior sampler, check a simpler possible
confound: distinct questions about one paragraph can have correlated question
and answer widths. Row/column offset cancellation in a paired objective does
not remove interactions between these statistics.

Use all 3,793 already admitted V63 training pairs and their unique training
questions. Verify the pinned pair report, source text, native cache report and
every accessed packed image. Recover question pixels by the exact prepared
prompt layout. Never use answer location in the paragraph as a predictor.

Two predeclared probes predict physical answer width:

1. Question width alone.
2. Question width, question ink fraction, and normalized horizontal/vertical
   ink centroids and variances. These summarize the image; they include no
   glyph identities, language embeddings or learned image encoder.

Use ridge regression with fixed penalty one after training-fold feature
standardization and target centering. Use five folds assigned by a stable
hash of the original training article title. Every question from an article
belongs to one fold. Fit on the other four folds, and evaluate the admitted
same-paragraph pairs in the held-out fold. This remains internal training-data
cross-validation, not a new language benchmark. No hyperparameter or feature
selection follows its scores.

For predicted widths r0, r1 and measured widths a0, a1, compare the summed
squared errors of correct and swapped assignment. Report strict correct,
wrong and tied pairs, and accuracy giving a tie one half. Also report the
uniform-context-weighted accuracy, because production samples contexts
uniformly before pairs. A fixed constant predictor must tie every pair.
Use tolerance 1e-10 in squared-cell-width energy units for numerical ties.
Persist each out-of-fold prediction and pair score, feature definitions,
fold membership, source hashes and resource scope.

Also count pairs with equal physical answer width and equal native suffix
width. Report overlap between these strata without changing admission.
Do not tune, rebalance or restart the active experiment from this audit.

Accuracy above one half would show that a simple feature interaction can
resolve some assignments. It would not prove that V63 uses that interaction,
that the features contain no task information, or that the actor can write
correct answers. Chance accuracy would not rule out other shortcuts. The
registered autonomous-writing and blank-question gates remain authoritative.
