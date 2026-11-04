# Raw-pixel beam decoding V56: frozen inference experiment

Date: 2026-11-05. Written before implementation or scoring new outputs.

## Question and mathematical contract

V55 learns a normalized conditional image density and improves known-form
generation. Local greedy decoding produces 98.4% known forms but only 7.4%
exact observed next images on its original 256-window audit. Pixel-wise
greediness need not maximize the complete image probability. Test whether
bounded search through the same learned density improves continuation.

    log p(X | H) = sum_p log p(X_p | H, X_<p)

At each binary pixel, expand every retained partial raster with ink 0 and 1,
add the exact Bernoulli log probability, and retain the eight highest-scoring
partial images per prefix. At completion, emit the highest-scoring surviving
image. There is no length normalization because every image has 1,024 pixels.
No image inventory, OCR, classifier, discriminator, external model, target
score, post-processing or extra objective enters decoding. The bank exists
only after generation, as an evaluator.

This is standard beam search applied to the raw-pixel factorization, not a
claim of algorithmic priority. Approximate mode search need not improve
meaning, and MAP-like language decoding can be repetitive. See
[Holtzman et al.](https://arxiv.org/abs/1904.09751) and
[Meister et al.](https://arxiv.org/abs/2010.02650). Report surface quality,
diversity and feedback as well as complete-image likelihood.

## Fixed model and implementation checks

- V55 joint EMA, SHA-256
  `ae4598c609648ba92a4eaa8680fa0842966a1f11b0757ed4a14b83038c048394`.
  No training, changed weights, dropout, precision change or parameter addition.
- FP32 evaluation with the same high matmul precision as V55; one physical
  RTX 4090 GPU 0. Prefix batch 16, beam width 8, memory <18 GiB.
- Shifted-row transformer is evaluated at each new row for retained beams.
  Within a row, gather its condition along surviving beam parent indices;
  current-row pixel branching cannot change that row's condition.
- Use stable log-sigmoid probabilities. Start one beam at log probability zero
  and the other seven at negative infinity. Preserve complete raster prefixes
  under parent gathering. No returned beam may be a bank image substitution.
- Beam width one must equal the original greedy decoder. Accumulated scores
  must match exact teacher-forced image log probability. A reduced binary
  enumeration test must recover the exact mode when the beam is exhaustive.
  Explicitly test a distribution where local greediness misses the global mode.

## Locked development audit

Use 2,048 deterministic public development windows, seed 20265620, with 64
visible cells and eight continuation cells. Source splits and renderer remain
unchanged. This is reused development data, not independent sealed evidence.
No training or sealed records enter the new evaluation.

Generate width-8 ordered, width-8 earlier-shuffled, and original width-1 greedy
ordered outputs on every window. Shuffle only the earlier prefix, retaining
its last four visible cells. All calls precede evaluator-bank construction.
Primary correctness is exact binary target-image match; nearest-image identity
is descriptive, not the primary statistic. Record self NLL, actual model NLL
of the returned image, exact-known-form membership, ink/blank/isolated-ink
statistics, distinct outputs, timing, memory and unselected raw galleries.

For the first 128 prefixes, generate eight visible-feedback steps under
ordered beam, initially earlier-shuffled beam, and ordered original greedy.
Each route rereads its own generated pixels. Shuffle once at the initial
prefix, not again during feedback. Report exact target-image match by step,
exact-known-form fraction, adjacent repeated-image fraction and distinct
images per stream, together with the actual target stream's repetition.

Supplied-image motor control: first 256 targets, zero hidden state and exact
observed image field, decoded by the same width-8 implementation. Record exact
pixel matches. This control reveals the answer and is not language evidence.

Use 2,000 source-record bootstrap draws, seed 20265719, for paired accuracy
differences. No beam-size, threshold, length, temperature or prompt sweep.
Save all per-example correctness, scores, source identifiers and pixels.

## Bounded decision

Retain this decoder for the next language experiment only if all hold:

1. width-one equality, enumeration, causality and score consistency pass;
2. model/source/corpus hashes, image-only boundary and resource checks pass;
3. ordered beam exact target match exceeds greedy by at least 2 percentage
   points and the paired source-record 95% lower bound is positive;
4. ordered beam exceeds earlier-shuffled beam by at least 1 point with a
   positive paired lower bound;
5. at least 95% of primary beam outputs exactly match known canonical rasters,
   blank rate <2%, and at least 99.5% of motor targets copy exactly;
6. ordered beam feedback target-match mean is no worse than greedy and its
   adjacent-repeat fraction is at most the actual-target fraction plus 0.10.

Even if retained, this is a bounded decoder improvement, not a completed ILM,
general fluency, instruction following, historical-form competence, new
language-model paradigm or efficiency parity. Preserve all V54/V55 decisions.
Failure should identify whether search, contextual choice or feedback remains
limiting, without changing the registered criterion after seeing the result.
