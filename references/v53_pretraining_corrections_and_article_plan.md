# V53 pretraining corrections and article plan

Date: 2026-11-04

## Pretraining corrections

No V53 optimization or development scoring has run. Inspection found that
`_visual_fields` calls V42 `encode_unit`, whose binary threshold cuts gradients.
The intended field and order objectives therefore cannot train the denoiser.
Replace that call with the same orthonormal transform applied directly to the
signed floating image, followed by normalization. Retain the threshold only in
evaluation and output. Test gradient flow from the field/order terms alone.

The original four-strata schedule always exposes one easy future cell. Sampling
exposes four terminal-noise cells. Alternate the registered stratified updates
with synchronous updates: draw one level uniformly from 0--99 per strip and
apply it to all four cells. This provides matched training support for the
synchronous inference schedule. Freeze this amendment before any training.

An all-dropped classifier-free batch is legal. Its order term is zero; it must
not throw or force a conditioned row. Resume must restore optimizer, EMA,
training generator, global CPU/CUDA RNG, dataset index and scheduler position.

Transferring V51 epsilon-prediction weights does not produce an already trained
velocity predictor. Record transfer as initialization only; all transferred
layers are fine-tuned under the new objective.

## Evidence and manuscript plan

The current 96-page manuscript mixes experimental history with the central
scientific argument. Preserve it as the long-form research record. Create a
compact article at `publication/ilm-nature/manuscript.tex` and separate
`supplementary_information.tex`, following the typography and package
organization of the inspected Lamp+LCD V-SPICE v13.7 template. Use the same
two-column article, wide summary and full-width numbered evidence plates,
without importing optical content, authorship or unsupported claims.

The article will explain image-input/image-output language modelling, the
conditional-distribution problem, and direct measured reader/writer results.
The supplement retains protocols, splits, numerical gates, compute receipts,
and the relation to earlier rejected mechanisms. V53 is prospective until its
locked report exists. Any explanation of V52's generation failure remains a
hypothesis: a vanishing terminal guidance term is true of a correctly learned
conditional diffusion too and does not by itself prove a causal failure.

Allowed files: V53 model/loss/trainer/evaluator/tests, the pretraining amendment,
new compact article package, pinned public evidence summaries, editable figures,
build scripts and Nutstore ILM delivery. Preserve unrelated files and old PDFs.

Validation: analytic velocity inversion, differentiable field regression,
split/RNG/hash checks, exact interrupted-versus-continuous smoke training,
single GPU run, paired-noise generation controls, unselected image gallery,
record-grouped bootstrap intervals, complete PDF build and visual inspection.

A paper layout is not evidence of a useful language model. No claim of Nature
acceptance, general instruction following, Qwen parity or superior compute
efficiency is permitted without an appropriate benchmark.
