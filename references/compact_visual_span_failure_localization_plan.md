# Localize the compact visual-language failure

## Decision and scope

Both completed compact fits fail the fixed language evaluation. The wider
writer retains full tested output rank but gives no complete exact QA answers.
Preserve both checkpoints, source files, outputs and scores. The next work is
a diagnostic comparison using these frozen weights; no new fit is selected.

The measured failure is unreadable, repetitive writing. Its cause is not yet
localized. Separate the following questions before changing the model.

| Diagnostic | Held fixed | Intervention | Measurement and interpretation |
| --- | --- | --- | --- |
| Codec control | Actual target ink and fixed codec | Decode the encoded reference glyphs directly | Pixel/foreground fidelity bounds what this codec can express for the diagnostic samples; this supplies the answer and is not generation. |
| Visual access | Visible prompt, frozen body and image encoder | Compare learned grouping with reference boundaries/counts, and compare image updates with native training targets | Feedback error, count/boundary errors and contextual-state differences test access and alignment; none is a language score. |
| Conditional writing | Correct reference-image prefix/history and trained writer | Supply reference lengths, sample the next span, then compare with shuffled contexts using identical noise | Latent/pixel error and context sensitivity locate writing failure before autonomous history drift. Reference lengths/history are explicitly supplied. |
| Numerical integration | Same weights, context, output length and initial noise | Compare the registered eight Euler steps with a prospectively fixed finer integration | A change in actual glyph quality isolates sampler sensitivity; a lower velocity loss alone cannot establish it. |
| Generated history | Same prompt, seed, writer and output budget | Compare reference-image updates with actual emitted-image updates at matched boundaries | Divergence after a usable first span tests accumulated feedback error; failure before feedback cannot be attributed solely to drift. |

Native embeddings may be used only as a labeled diagnostic bypass of the
image encoder. They introduce information and a distribution change relative
to the deployed visual path. A successful bypass does not qualify visual
reading, and a failed bypass does not uniquely identify a writer defect.
Never merge supplied-history results with autonomous answer counts.

## Registration before execution

Bind the actual checkpoint, codec, source, case identities and original pixel
hashes in a machine-readable request before running model calls. Fix sample
selection, integration counts, seeds, metrics and runtime ceiling in advance.
Include seen training examples and separate development groups to distinguish
failure to fit from failure to transfer. Keep these diagnostic samples outside
any later untouched confirmation set. Retain every selected failure.

Implement the interventions in a separate diagnostic entry point. Preserve the
authenticated V1 and wider model sources and deployment loaders. Reuse the
existing corpus, teacher cache and foundation. Test exact context/target index
alignment and matched noise before interpreting the measurements.

Run serially under the project GPU mutex on GPU0 after the shared-workstation
admission checks. Record actual allocated memory, elapsed time, waits and
process ownership. The diagnostic request must declare its case/call budget
and deadline before launch; this plan is not a completed registration or an
executed experiment. Do not start a competing training job.

## Decision after measurement

Choose one intervention from the earliest measured failure. Register a matched
training comparison only after this localization. Re-evaluate full autonomous
answers and paired-question dependence after any repair, retaining the
unchanged failed control. Development improvements need fresh source groups
for confirmation. The full goal also requires historical and unencoded forms,
usable task quality and measured consumer-GPU costs.

## Fixed implementation specification

`scripts/diagnose_compact_visual_span.py` prepares an immutable request on CPU,
then runs the two final checkpoints serially. Select eight QA and eight book
records from each of the training and development partitions: 32 records in
total. Sort source groups and records by the fixed namespace hash and take
one per group in rounds. The two development works each contribute four book
records. These are diagnostic development samples, not a fresh confirmation
set. For QA, the full question is the prefix. For books, use the first half
of the reference spans. Test at most eight following spans per record and
retain all selected targets and failures.

Use seed namespace `compact-visual-span-failure-localization-v1` and base seed
202611156, deriving a separate fixed seed per case/span. Six arms share the
exact initial noise at every target span: reference-image context with eight
or 64 Euler steps, native-embedding reference context with eight steps, a
wrong reference context from the next selected record with eight steps,
emitted-image history with supplied update counts, and learned prompt grouping
with emitted-image history and learned update counts. All arms receive the
reference next-span length. This is reconstruction under diagnostic inputs;
it does not produce new autonomous language scores or test learned stopping.

Compare target latents and original ink, foreground F1, exact glyph counts,
context sensitivity, feedback error, update counts and boundaries. Direct
codec decoding is the supplied-target positive control. Verify the causal
reference-prefix state against the full causal reference-history state before
interpreting the history comparison; require relative RMS difference at most
0.02. Check and retain actual initial-noise hashes and emitted-pixel feedback
hashes. Save raw PNGs, binary ink, continuous endpoints and per-case metrics.

Before neural execution, check that every selected and training-used native
unit has the same vocabulary row string in the compact and teacher archives.
For each trained encoder, measure the closest point in its fixed feedback
head's affine family. With final LayerNorm, let

\[
 A=D W\operatorname{diag}(\gamma)C,\quad
 a=\mu+D(W\beta+b),\quad C=I-\mathbf1\mathbf1^\top/m.
\]

For target embedding \(e\), the squared distance to
\(a+\operatorname{col}(A)\), divided by \(\|e\|^2\), is a lower bound
for that head's relative error in exact arithmetic. Compute the projector
from the actual float64 SVD. Aggregate uniformly over the selected source
positions. This relaxes the preceding network and does not measure the full
training objective's batch weighting or imply a language-quality bound.

The run makes zero optimizer updates. Its deadline is 1,800 seconds with at
most 300 seconds waiting on pressure. Reuse the existing full-reader admission
of 48 GiB host RAM, low memory stalls, 20 GiB free GPU memory, the exclusive
GPU0 mutex, a 24 GiB runtime host reserve and a 22 GiB allocation ceiling.
This is the same frozen body/codec workload as the measured fits and inference;
no other project's process is changed. Bind the implementation and exact
case/call budget before launch. Complete and interpret the recorded comparison
before registering another fit.

## Prepared request

The [CPU request](evidence/compact_visual_span_diagnosis_registration_20261111.json)
binds 32 actual records and 201 target spans, with a maximum of 41,808 velocity
calls and 1,124 contextual-body calls across both final checkpoints. No neural
model call was made during preparation. All 26,733 training-used or selected
native vocabulary rows match between the two archives. Each feedback head's
centered affine map has rank 383 in 2,048 output dimensions. The
[per-record geometry](evidence/compact_visual_span_feedback_geometry_20261111.json)
is retained separately. These capacity/target-error measurements do not
complete the neural diagnostic or identify the sole language failure cause.

## Causal-cache execution amendment

The first registered neural attempt stopped after 125.07 seconds at its fixed
prefix consistency check. It retained 25 completed V1 case reports and four
reference arms of the following book case; the wider model had not loaded.
Do not treat the partial run as a completed comparison or alter its outputs.

A separately registered one-case probe made twelve body calls and 24 velocity
calls. With exactly identical prefix embeddings, BF16 full-sequence versus
prefix-only context has 0.050291 relative RMS difference; FP32 arithmetic on
the same BF16-rounded weights reduces it to 0.000002592. Erasing future inputs
at the same sequence shape is bit-equal at the tested prefix in both dtypes.
The prefix mask/cache flags are also bit-equal in the BF16 comparison. Tiny
codec/encoder batch differences are measured separately. This identifies a
shape/precision-dependent numerical discrepancy for this case, without proving
that it explains the language failure or all other prefixes.

For the replacement diagnostic, compute reference contexts through the same
prefix prefill and cached updates as generated-image history. Correct reference
images are fed one span at a time with supplied counts; the native bypass uses
the matching native prefix and cached updates. Retain the full-record reader
metrics separately. Keep cases, checkpoints, seeds, eight/64-step sampling,
other interventions, metrics and the original 0.02 prefix consistency tolerance.
Do not relax that tolerance. The body-call ceiling becomes 2,056; the velocity
ceiling remains 41,808. Reuse the exact authenticated native vector payload
through a hard link after value equality checks. The stopped attempt and
32.51-second precision probe remain in total diagnostic cost reporting.

The replacement's [cached-execution request](evidence/compact_visual_span_diagnosis_cached_registration_20261111.json)
retains all 32 cases, 201 target spans, checkpoint identities and matched seeds.
Its native-vector file shares the original authenticated inode. The full
[precision probe](compact_visual_prefix_precision_result.md) records the
stopped attempt's cause investigation. The replacement's language outcome
remains unmeasured until the whole diagnostic is complete and scored.
