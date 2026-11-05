# Reuse a faithful output representation instead of inverting the input encoder

5 November 2026. Registered before this 128-case codec audit. V66's direct
writer final audit remains fixed. No V66 threshold, source or sample changes.
The input encoder's semantic coordinates do not have to be the output
renderer coordinates. The previously qualified V34 codec offers an existing
continuous representation with measured modern/historical reconstruction.

## Question and limits

Can that fixed codec preserve the same Chinese/English physical blocks used
by V64/V66 without learning a new inverse of the Qwen visual encoder? This is
an output-interface qualification, with target images supplied to a frozen
encoder during reconstruction. It does not test prediction of unknown answers,
semantic understanding, language generation or unseen-font generalization.
The 64 fitting images used by V65 are already observed and not this audit.

## Immutable inputs and procedure

Use the verified V34 update-6,000 EMA checkpoint, SHA-256
`a138c9cb3b0502e43d1227f689c020893d56b468742c32e1840e44d299662f33`.
No optimization, selection, denoising schedule, font rendering, glyph bank,
character inventory, OCR or model-service call is permitted. Native codec
input is white=1, black=0. Preserve the V65 polarity failure separately.

Read the original V64 fixed 128-case development manifest and the matching
complete-image cache report SHA
`14b088cff0c91175570f4635af7363419225cb36c8784d11efa679b0993bb369`.
Manifest SHA-256 is
`16646a5e8379a991d40345e9a2b844604feccf2bdf6edbc77d159c71f4ec3934`.
The manifest/cases are reused development, not a fresh generalization test.
For each exact physical 32-by-64 image, split left/right 32-by-32 tiles, encode
each to its native normalized 768-vector, and decode the vectors jointly as
the two halves of one output block. The `CodecBlockWriter` accepts only the
continuous vectors during prediction; it receives no target image there.

Generate all 128 blocks once under correct continuous codes and once under
the same one-case cyclic code permutation used in V64/V66. Use the native
codec logits' zero threshold (equivalently white probability .5). Retain all
256 raw generated images, all 128 original reference images, per-case scores
and first eight fixed examples. No output repair, rejection or reselection.

Keep the existing interface gates: mean foreground F1 >=.90 and correct-minus-
shuffled mean foreground gain >=.20. Report exact blocks and blank fraction.
Both must pass. Passing qualifies this fixed renderer/latent representation
only, not the context-to-latent predictor. The decoder has no conditional-choice
or instruction capability by itself.

## Execution and next actor

Entry point: `scripts/audit_codec_output_v67.py`, CPU only, four threads. Run
after V66's owned training process terminates, with no new GPU model job.
Verify original cached source images and the checkpoint; bind source hashes,
reference/condition/generated images and numerical decisions. Preserve the
original V66 final result whichever way it goes. Commit/push this protocol
and audit implementation before generating audit outputs.

If qualified, the next complete actor can retain the inherited visual reader
and contextual body while predicting two output-codec vectors per physical
block. The output head maps continuous contextual states to the codec space;
detached codec encodings supply training targets. No input-encoder inversion
is required. Recheck actual full-model memory, causal/cached execution, exact
optimizer replay and frozen-decoder gradient flow before language training.
Old V35/V47 failures show that a good codec and latent regression do not
guarantee language. A real prompt-dependent training/audit remains required.
