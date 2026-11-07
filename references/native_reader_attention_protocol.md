# Adapt contextual attention while preserving learned visual features

The completed visual-only dense checkpoint copies 13/16 Chinese and 16/16
English strings on its original 1024-by-256 copying strips, but answers
0/32 Chinese and 0/32 English natural questions. The later coarse/fine
state-alignment learner improves English but loses Chinese copying on its
square/native-view interface. These are different recipes and geometries;
their difference is not a controlled attribution to one component.

This learning intervention starts from the exact completed dense checkpoint
`016429ec5454cdd681693d881df4c6b7d52fb43dcae7b9e15857d2d9cd32b483`.
Freeze its complete visual frontend, including the learned visual adapters.
Keep its original geometry: complete copying strips and complete square QA
pages. Do not pool, resize, crop or replace the visual features. The choice
of this geometry is fixed before inspecting the new canvas-control outcomes.

Add fresh rank-16, alpha-16, zero-output LoRA residuals only to q/k/v/o
projections of the 28-layer native language body. Original source weights,
input/output embeddings, language head, image generator and codec remain
frozen. This tests whether contextual attention can learn the question-to-
answer mapping without relearning an already partially functional glyph
frontend. It is a transfer-learning hypothesis, not a new foundation algorithm.
Zero output must reproduce the source's native decisions before optimization.

Reuse all 640 original bilingual training IDs: 128 copies, 256 Chinese QA
and 256 English QA from article-disjoint paragraphs. The copying strips
must reconstruct their original square inputs exactly when white bottom
padding is restored. Prepare frozen native image prefixes offline, using
the original per-record seeds. All other training input pixels stay exact.
Cache no annotated answer inside a visual prefix; no transcript or text-state
teacher conditions the model. The source training JSON also contains unused
transcript fields, which are parsed but never passed to a text pathway.
The same source vocabulary head supplies training
cross entropy and a temporary diagnostic readout.

For a visible prompt image X, fixed image prefix E(X), correct answer y,
and low-rank attention parameters A, optimize
L = -mean_j log p_U(y_j | H_A(E(X), y_<j)), including native EOS.
Correct history enters only teacher-forced training. Native causal masks
must block future answer history from every prompt-prefix state. There is
no semantic-state MSE, image corruption, alternative-answer lookup or target
image supplied at inference.

Run four independently shuffled complete passes over all 640 records,
exactly 2,560 updates. AdamW uses betas (0.9,0.95), decay 0.01 and global
gradient clipping at one. Warm up to 5e-5 over 64 updates, then cosine decay
to 5e-6. Keep BF16 body computation and FP32 residual masters/losses. Use
gradient checkpointing, batch size one and one GPU. Do not select a checkpoint
or change coefficients from validation; assess only the final state.

Before the main run, execute three actual updates on the largest prefix,
largest answer and first Chinese copying record. Require all attention
residual tensors to have finite gradients, nonzero gradients by the third
update, changed initially-zero up matrices and an exact disabled-source
logit replay. Replay four fixed source copy responses both with disabled
residuals and zero enabled residuals, against the original decisions.
Verify the future-history mask and a temporary FP32 pre-answer-state check
on the first copying record, tolerance 2e-5 after RMS normalization. A failed
preflight stops the main optimizer. Start the main run afresh afterward.

Retain all 2,560 optimizer records, four passes per ID, optimizer/RNG state,
source/data/code hashes and atomic checkpoints. An interrupted job may resume
only after terminal exit, from the same saved state with its complete prior
journal copied through the checkpoint update; never restart a live process.

Assess all 160 original legacy canvas-control conditions, with the unchanged
input hashes, seeds, normalization, annotated alternatives and 96-decision
top-one/EOS readout. The legacy baseline must have replayed all 160 responses
exactly before its use. Require the same six reader prerequisites as the
preceding state-alignment study: at least 30/32 copies; 16/32 Chinese answers;
4/16 both-correct Chinese pairs; a Chinese gain of at least four; at least
11/32 English answers and 3/16 both-correct English pairs. The English quality
floor is the real frozen native-432 source, not the failed full-page baseline.

These gates only admit a separate continuous image-writing experiment. The
temporary Unicode head, externally pretrained complete model and visual
frontend remain explicit. Passing them cannot qualify an independent
image-input/image-output ILM, historical-form generation, from-scratch single-
GPU pretraining or efficiency relative to a conventional LLM. CMRC 2018 and
SQuAD 1.1 retain CC-BY-SA-4.0 attribution to Cui et al. (2019) and Rajpurkar
et al. (2016); Show-o2 source/model are NUS Show Lab, Apache-2.0.
