# Repair visual access while preserving the native generator

Specified before optimization, 7 November 2026. The complete HQ diagnostic has
20 image-reading calls, eight text-supplied writing controls and twelve
subsequent plain-text calls. Plain text recovers all eight requested copy
contents, although four Chinese responses retain quotation marks; the same
body invents unrelated strings from their images. All original calls remain
immutable. This supports testing visual access; it does not identify its unique
cause or qualify an image-language model. Chinese writing also remains poor.

Use the same complete native model and codec. Adapt only the semantic visual
encoder and fusion linear layers with rank-16, alpha-16 residuals. Keep all
pretrained weights and the language body/head frozen. FP32 adapter masters and
AdamW states use BF16 native computation. Gate residuals on only while forming
the clean source-image prefix; the native generator's base path stays available
with the gate off. Keep the gate on through checkpoint recomputation/backward.
Verify zero-residual prefix equivalence and unchanged disabled-gate output after
real updates. This source-reading stage is an intermediate prerequisite for the
image-answer path, not a replacement objective or a deployed ILM.

Use all 128 training copies and 256 training QA examples already prepared, with
32 copy and 32 QA validation examples excluded from optimization. Source pages
and answer strings are unchanged. Cache the native FP32 VAE posterior samples
once, with no resizing or answer-aware crop. Use seeds 20268921+i in original
448-case order, except the twelve existing diagnostic IDs retain their original
20268521+j latent seeds. Authenticate those twelve latent tensors against their
original observations. Store complete source and reference hashes.

For raster X and target text y, teacher-force only preceding y decisions and
minimize the frozen native language-head cross-entropy. All changing prompt
content enters through source pixels. Correct answer histories are available
only in this training loss; autonomous evaluation receives neither histories
nor answer strings. This explicitly uses Unicode/token labels as auxiliary
training supervision. It is not a claim of vocabulary-free training. A later
image-output runtime must remove its language lookup/head after compiling fixed
structural controls and must demonstrate correct autonomous answer images.

First run two real optimizer steps on the longest training copy and the largest
training QA prefix. Check finite/nonzero gradients, actual residual changes,
native prefix equivalence at zero and unchanged disabled-gate prefix after
optimization. This is an execution check, not reading or language qualification.
Then start fresh residuals for 4,096 steps, batch one, seed 20268921, alternating
copy and QA from independently shuffled cyclic training lists. AdamW uses
rate 1e-4, betas (0.9,0.95), weight decay .01 and clip one, with 100-step warm-up
and cosine decay to 1e-5. Save source/data/code-bound resumable checkpoints.

Autonomous reading checks must include the fixed twelve-case cohort before
and after learning at matched BF16 computation, followed by all 64 held-out
copy/QA cases if the screen improves. Report exact canonical stopped strings,
raw responses and alternative correct wording separately. The necessary copying
requirement remains 30/32 requested strings alone. Natural QA, question-erasure
interventions, paired-question consistency, Chinese image writing and efficient
independent deployment remain required for the full goal. No text-helper score
can mark that goal complete.

Use native incremental KV decoding and last-position logits only after its
complete generated decisions match all twenty original FP32 reading calls.
Preserve any differences. Numerical cache verification is execution evidence;
it does not improve incorrect source answers or establish language efficiency.
