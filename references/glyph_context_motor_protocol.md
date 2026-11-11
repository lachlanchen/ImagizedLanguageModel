# Actual-context glyph motor: fixed training and sampling protocol

This implements the [bounded decision](glyph_tail_context_writer_decision.md).
The original actor remains the control. Its reader, contextual adapters,
encoder E, decoder D and stopping are fixed. Only a new continuous conditional
motor distribution is trained. Earlier semantic-flow and balanced-writer
failures remain evidence against assuming that this will improve language.

## Model and data

Condition directly on actual 4,096-coordinate reader states from the
authenticated balanced-writer cache: 3,176 first and 20,086 later states.
Check both label orderings against all 3,176 original training jobs. The
source cache report and original parent checkpoint remain hash-bound.
No diagnostic/development/native-helper state or answer enters training.
Primitive images are training-only targets, never a deployed lookup table.

Obtain target motor features with the frozen current parent:
`decoder_input(E(1-2*ink))`, shape 64-by-4-by-4. Calculate all target features
on CPU FP32 in batches of 32. Compute normalization with equal total mass on
the first and later position pools, uniform within each pool. For each motor
coordinate use its mean and standard deviation, clamping scale below at 0.05.
Retain normalization and target features with source hashes. This is supplied
training-target preparation, not an autonomous language result.

The velocity network has a 1,024-to-1,536 noisy-input linear, LayerNorm4096
and a 4,096-to-1,536 context linear, sinusoidal128 time features followed by
128-to-512-to-1,536 with SiLU, three width1,536 residual blocks, and a
1,536-to-1,024 output. Each residual block is LayerNorm, linear, SiLU, linear
and a 0.1 residual scale. The output starts at zero. There is no final narrow
projection below1,024 coordinates. This capacity statement guarantees neither
successful training nor language. Record the actual parameter count.

For standardized target y, independent standard Gaussian epsilon and uniform
t in[0,1], use x_t=(1-t)epsilon+t*y and unweighted mean squared error against
velocity y-epsilon. This is conditional flow matching; the
[primary formulation](https://arxiv.org/abs/2210.02747) does not establish
finite-model ILM correctness. No auxiliary semantic label loss, pixel reranker
or synthetic semantic mixture is added.

## Fixed exposure and numerical execution

Seed202611171. Train exactly12,000 batch128 updates:64 uniformly sampled
first positions and64 uniformly sampled later positions each update. This
gives768,000 exposures to each pool. AdamW lr2e-4, betas(0.9,0.99), weight
decay0.01, gradient norm clip1,200-step warmup then cosine to10%of peak.
Use FP32 parameters/targets/loss and BF16 linear compute, with the repository's
fixed numerical backend. EMA decay min(0.999,(1+step)/(10+step)); only the final
EMA is eligible for evaluation. Checkpoints every1,000 updates retain model,
EMA, optimizer and all RNG states. Never choose an earlier checkpoint on
development results. Final small-head parameters and preserved modules are
audited separately; inherited pretraining and cached-reader cost are not
included in the small-head training cost.

Use a maximum1,800 wall seconds per fit attempt including at most600 seconds
pressure waiting. A failed/interrupted attempt must remain; any resume needs
authenticated checkpoint/RNG/exposure and a separate recorded attempt rather
than an unrecorded restart. Do not automatically extend the exposure or retry
after a negative result. GPU0 mutex,48GiB available host entry/24GiB runtime,
20GiB free GPU admission,8GiB peak allocation for this small-head fit, and
PSI some<=0.5/full<=0.1. CPU target preparation uses two threads,600 wall
seconds,32GiB host entry/24GiB runtime,4GiB peak RSS and CUDA hidden.

## Frozen language comparison

Bind all216 existing physical inputs before fitting:96 fixed-cell QA streams,
24 fixed-cell book/control streams and96 proportional QA streams. The runtime
manifest contains image identities and opaque cases, not answers. Keep the
reader, E/D and learned stop unchanged. No reference length is supplied and
the cap stays128 glyphs. Per-case seed is the first eight SHA-256 bytes of
`glyph-context-motor-v1\0` followed by the case ID, interpreted big-endian and
reduced modulo2^63. The seed never depends on an answer, score or sampled image.

Generate one Gaussian motor sample per glyph with32 Heun intervals, using
Euler for the last interval. Decode through the frozen neural D, threshold
at0.5 and feed only the actual emitted pixels back through E and its preserved
semantic affine transform. Record noise, sampled motor features, actual pixels,
states, stop probabilities and feedback hashes. No lexical table, old
deterministic semantic point, OCR, teacher, font, prototype bank, reference
search, best-of choice or sample repair participates. The ancestor actor also
computes a prediction when it reaches its output cap; retain and identify
that un-emitted final motor sample rather than counting it as emitted text.

Evaluate the final EMA once under the unchanged complete-response rules.
Review every original output before opening references for scoring, disclose
prior cohort familiarity, independently recount all metrics, and retain all
failures. The prospective development pilot is>=8/32 proportional exact,
>=2/16 both-correct pairs,>=11/32 readable, retention of all six earlier exact
proportional answers and both exact fixed answers. Always report erased
controls, all twelve normal book continuations, stop/cap counts and costs.
A pass needs new untouched source-group confirmation. It does not establish
general language, model novelty, one-GPU large-model parity or efficiency.

The full-reader stage retains the decision's3,600-second per-attempt ceiling,
600-second pressure allowance and22GiB allocation cap, one body loaded
serially, foreign GPU services unchanged. No manuscript claim is updated from
training loss alone. The full independent usable-ILM goal remains active.
