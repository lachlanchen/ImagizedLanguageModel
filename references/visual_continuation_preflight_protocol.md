# Continuation-only adaptation: real single-GPU preflight

This preflight tests execution, causality, gradients and memory, not language
quality. The correct-image-history result and all previous language scores stay
unchanged. The full modern-Chinese, book, historical and unencoded ILM goal stays
active. A successful preflight is a prerequisite to a separately registered fit.

## Fixed candidate

Use the authenticated 36-layer source image reader with its original zero-output
adapters, without importing the older four trained tail adapters. Prompt image
prefilling calls this frozen source directly. Copy its detached key/value memory
once per training case. No answer-layer state cache is reused.

After the prefill, append one trainable continuous query initialized from the
last fixed numeric task-control vector. At each of all 36 answer layers, apply
an additive rank-eight residual with alpha sixteen and zero-initialized up map.
Only the answer computation calls these residuals. Continue from preceding glyph
images encoded by the frozen existing neural image encoder. Recompute all 36
answer layers with nonreentrant activation checkpointing. A stateless cache view
concatenates frozen prompt keys/values with the current answer keys/values,
without appending anything to the prompt during forward or recomputation.

Initialize the neural readout and stop predictor from the frozen balanced
checkpoint `dcac4ee207614adca78a8b99cc4fe422b3197204d9028192bef30f111d61586d`.
Train those two heads, the query and all new residuals; keep source vision,
source context, encoder, decoder and normalization frozen. Source vocabulary
matrices, Unicode IDs, OCR, fonts and teacher services do not enter model execution.

## Data and four updates

Select only from the verified 13,785-sequence training split. Exclude the earlier
feedback-ceiling diagnostic controls and their source groups, and current
held-out QA groups. Choose four distinct cases deterministically: shortest QA
answer, largest QA prompt, longest QA answer and longest book passage. Resolve
length ties by source ID. Supply the first 128 original glyphs of the selected
book as visible prefix and train on its complete remaining continuation. Preserve
complete QA prompts and answers. Do not truncate targets for memory admission.

Pack prompt glyph images into rows of 32 original 32-square tiles, with white
padding only. Apply the unchanged native physical observer. Registration binds
all selected source/raster shards, prompt images, target arrays and runtime
source versions. The old 3,176-question adapter subset had answers of at most
16 glyphs; it is not the data-capacity test for this candidate.

Make exactly four AdamW updates, one on each case in the order above, seed
2026111217, learning rate 0.0001, no weight decay, full FP32 trainable parameters
and optimizer state. Use the unchanged balanced physical reconstruction loss
on all target glyphs plus 0.1 times balanced stop/continue BCE. Include one
terminal boundary. Retain per-step gradient norms and losses; finite loss and
four updates do not demonstrate learning. Discard these trained parameters
before any full comparison; preserve their checkpoint as preflight evidence.

## Required checks

- CPU tests on actual native decoder layers: pure-cache recomputation, native
  zero-residual equivalence, causal masking, checkpoint/plain gradient agreement,
  incremental/parallel inference agreement and absence of source gradients.
- On the full source body, require the zero-residual split answer states to agree
  with the native concatenated computation within max absolute error 0.002.
  This tolerates measured batch-shape floating-point differences; it is not
  bit-exact equivalence of different execution shapes.
- Verify finite nonzero up gradients in every one of 36 residuals on all updates,
  nonzero down gradients after the zero-initialization step, nonzero query and
  readout/stop gradients, and zero source/encoder/decoder gradients. Require
  every residual's up/down tensors and the query to change by the final update.
- Keep frozen parameter version counters unchanged. After the updates, recompute
  the original source prompt and require bit-exact complete hidden-state and
  per-layer key/value fingerprints against its pre-update execution.
- After the fourth update, change the last physical reference image. Require
  all earlier predictor states to be bit-exact. Change the first physical target
  and require the first query state to be bit-exact. Target changes must affect
  the later state, so a disconnected computation cannot pass.
- Compare parallel answer states with serial cached answer states on the short
  case after updates, tolerance 0.002. Retain a short actual-pixel-feedback
  inference invocation with learned stopping. An empty or malformed output stays
  visible and is not a failed language test or a qualified model.

Register sources and data on CPU, commit and push registration before GPU calls.
Run one worker under the existing GPU0 mutex. Allow 1,800 seconds total, including
at most 300 seconds of pressure waiting. Entry requires 48 GiB available host RAM
and 20 GiB free GPU memory; runtime reserve is 24 GiB host RAM and peak allocated
VRAM must stay at most 22 GiB. Keep GPU1 and foreign services untouched. Preserve
failed attempts and stop on a failed check; do not silently reduce layers,
sequence lengths or precision to turn a failed registered preflight into a pass.

All checks must pass before a full fit is registered. Otherwise record the
specific failure and change the hypothesis or register an explicit implementation
correction. A memory result applies only to these measured complete cases, not
unbounded pages, batch sizes, or matched-quality efficiency.
