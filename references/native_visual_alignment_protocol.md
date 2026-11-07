# Full-page supervision after failed visual question answering

Specified before the new optimizer runs. The predecessor's complete result is
27/32 exact stopped held-out copies through an auxiliary text head, but 0/32
correct natural answers. Question erasure leaves 29/32 normal outputs
unchanged. The previous goal turn made progress by completing that experiment,
the controlled assessment and the evidence publication; it did not qualify an
ILM. This new experiment does not change the full image-input/image-output goal.

## Hypothesis and fixed scope

Short answer supervision may be inadequate to align complete Chinese page
images with the retained language body. In the 256 training QA records, input
prompts contain 73,425 Han-character occurrences and 2,387 distinct Han
characters, whereas answer labels contain 1,418 occurrences and 628 distinct
characters. The 128 training copies directly supervise only 81 distinct Han
characters. Repeated passages occur in the paired-question records, so these
are record-level counts, not independent texts. Missing local labels do not
mean missing pretrained knowledge. These observations motivate a test; they
do not establish the unique cause of failure.

Keep the same source/model/codec, native prefix, gated rank-16 visual adapters,
full source geometry and cached FP32 VAE samples. Keep the language body/head,
generator and all pretrained weights frozen. Initialize from the completed
adapter checkpoint with SHA256
`e2937d2fa986fac0e7ca1964607d5dca06194b698b6a25ad0644635f585427fb`.
Discard its optimizer/RNG state. No new images, validation pages or answers
enter optimization. The new input supervision is the complete original text
already rendered into each training image. Reconstruct it from the original
preparation code and require its complete text SHA256 to match the original
page geometry metadata. Authenticate the original image/cache bindings.

For source image X, full-page transcription s and answer y, minimize the same
frozen-head mean cross-entropy, including EOS:

\[
 L_{\rm read}(X,s)=-|s|^{-1}\sum_j\log p_U(s_j\mid C_{\rm read},R_A(E(X)),s_{<j}),
 \quad
 L_{\rm answer}(X,y)=-|y|^{-1}\sum_j\log p_U(y_j\mid C_{\rm answer},R_A(E(X)),y_{<j}).
\]

The two task controls are fixed instructions. Changing passage and question
content still enters through pixels. Auxiliary Unicode/token labels and correct
answer histories remain explicit training aids. They cannot qualify the final
vocabulary-free image runtime. Dense transcription alone is not semantic QA.

## Budget and execution

First replay all twelve predecessor readouts exactly from its numeric adapters
and authenticated cached image latents. Then execute three real preflight
updates: the longest tokenized training QA transcription, the longest training
copy transcription and the longest training QA answer. Check finite gradients
through every adapter, actual parameter changes, frozen-base gradient absence
and unchanged disabled-gate prefix. Measure memory on these complete targets.
The full run starts again from the predecessor, never the preflight checkpoint.

Use 2,560 new updates, batch one, seed 20269221. The first 1,024 updates are
full-page transcription. The remaining 1,536 cycle transcription, QA and
requested-only copying. Independently shuffled cyclic pools yield exactly
four complete transcription passes over 384 training images, two answer passes
over the 256 QA records and four requested-copy passes over 128 copies. Save
the complete chosen-ID/task schedule before optimization.

AdamW: rate 1e-4, betas (0.9,0.95), weight decay .01, clip one, 50-update
warm-up and cosine decay to 1e-5 across the complete 2,560-update run. FP32
adapter masters/states use BF16 native computation. Preserve checkpoints every
128 updates with optimizer/RNG state and source/data/initializer/code hashes;
resume unchanged code into a fresh output directory. One GPU job at a time.

## Assessments and claim boundary

Retain before/after transcription readouts for the first four held-out copies
and first four held-out QA pages, with native top-one decoding, 1,024-decision
cap and the same input latents/seeds. References remain separate. Report
canonical exact stopped transcripts and order-sensitive character edit error;
never repair output or replace missing content. This screen is diagnostic,
not a replacement for answering.

Retain the same full 64-case held-out answer observation and all 48 paired
normal/erased-question calls after the complete run, even if the transcription
screen fails. Use the existing immutable input indexes and matched passage
seeds. Report copying, natural correctness and both-correct pairs separately.
All failures remain visible. Existing public development data are exploratory,
not a pristine sealed test. The predecessor and this curriculum have different
additional compute/target lengths; without a matched continuation arm, a gain
cannot be causally assigned to dense supervision alone.

The immediate gate is correctly answering held-out questions from their images,
with paired-question consistency and erasure controls. Reliable writing still
requires at least 30/32 correct requested copies. Passing transcription or
these temporary text-head checks cannot complete the ILM goal: direct native
image-answer learning, removal of the runtime vocabulary/head, correctness of
the actual generated images and capability-matched efficiency remain required.
