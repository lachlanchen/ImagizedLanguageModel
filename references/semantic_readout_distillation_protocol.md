# Offline semantic readout distillation: prospective protocol

Preserve the completed primitive, 508-question glyph loop and changed-date
checkpoint. This intervention changes only the existing continuous readout G.
It does not add language knowledge to the frozen contextual body. The runtime
still receives image pixels and produces image pixels, and re-encodes its actual
emissions. No source head, tokenizer, font, glyph bank or answer bank is deployed.

The completed target-coverage audit motivates a bounded transfer experiment.
Only 1,074 forms occur as positive language targets, against 4,087 forms in the
primitive. Coverage is a hypothesis about the writing failure, not its proven
sole cause. A random small head trained on 508 answers need not reproduce the
native output decision function. Distillation is a training technique, not a
new physical law; see [Hinton et al.](https://arxiv.org/abs/1503.02531).
[PIXAR](https://arxiv.org/abs/2401.03321) supplies prior pixel-language work,
not evidence for this intervention's effectiveness.

Preparation authenticates the existing shared source archive, primitive
inventory/checkpoint, completed baseline and all 508 causal-cache files. Only
training-question states enter. Whole question histories are split by their
fixed cache order: every fifth history is held out from this compiler fit.
No development state, answer or observed success selects calibration classes.
Map each source unit to its first complete decoded form when that form occurs
in the existing training-only inventory. Skip partial Unicode, empty or
unsupported starts; record exclusions. Source end markers receive a separate
terminal label. For each primitive form request its existing single-source-unit
anchor, retaining the canonical aliases chosen in the original preparation.
All source IDs remain offline labels.

The temporary full native output matrix W labels genuine states and validates
synthetic hidden-state calibration. The matrix is read from the verified shared
GGUF; no second weight file is saved. For target k and current full-head winner
j, use training-state diagonal variance S and the correction

    d = S (w_k - w_j)
    h <- h + max(0, 1 - (w_k - w_j)^T h) d /
                    max(1e-8, (w_k - w_j)^T d).

This is the minimum diagonal-Mahalanobis displacement to one linear half-space.
It is not a proof of convergence against all output rows. Recompute full-head
argmax after each step, at most 24 times; retain only samples whose *actual*
final teacher winner equals k. Record all requested/accepted counts, coverage
and standardized RMS displacement. These are counterfactual hidden vectors,
not new passages, books, valid natural-language contexts or semantic examples.
Generate four training bases and one held-out base per form; calibration bases
come only from the corresponding genuine-history split. Fixed seed 202611103,
batch 32, margin 1, FP32 full teacher arithmetic without TF32. No body is loaded.

Initialize from the preserved language checkpoint. Train G alone for 6,000
batch-128 AdamW updates at 2e-4 with the prior cosine schedule. Half each batch
comes from accepted training calibration states, half from supported genuine
training states labeled by the teacher. MSE to the normalized semantic anchor,
0.2 cosine discrepancy and balanced pixel BCE through frozen D are unchanged.
Keep E, D, affine buffers and stop predictor bit-exact. The terminal predictor
is retained, not distilled; terminal/unsupported teacher events are excluded
from this G-only fit. No full language-weight training or language-quality
claim follows from this fit.

First qualify this function compiler, before more expensive language inference.
Compare the preserved baseline and candidate on *all* accepted synthetic
held-out states and *all* supported genuine held-out states. Measure physical
pixel exactness, foreground F1 and fraction with F1 >=0.95. On both routes,
require mean F1 >=0.95 and at least 80% of outputs reaching F1 >=0.95. This is a
prospective renderer/function-transfer gate, not language accuracy. If it fails,
preserve the result and revise the readout rather than running another long
language job. If it passes, define a separate 508-question continuation and
actual image-feedback assessment; retain date behavior and erased controls.

Each stage has an 1,800-second cap, 8 GiB GPU cap, 48 GiB host entry reserve,
32 GiB runtime reserve, existing GPU0 mutex, <=0.5/0.1 some/full PSI and <=300s
pressure waiting. Inspect ownership before starting. These separate small-head
limits do not modify any completed body's or diversity run's admission rules.
Preserve foreign GPU services. Save status, timings, hashes, losses and every
qualified measurement; failure is a result. Do not claim efficiency parity with
the original baseline, since this adds teacher computation and supervision.
