# Learn visual reading states from the same frozen language body

This prospective learning experiment follows completed failed answer-only and
full-transcription adaptation. Chinese source images still give 0/32 exact
answers, while original text gives 16/32. Native-scale English source reading
gives 11/32. Full-resolution codec reconstruction preserves Chinese writing
far better than the native-square downsample. Source-window and pooled-context
diagnostics remain separate controls, including all their failures. Their
results cannot themselves qualify learning or direct image answers.

The new hypothesis is to align the image prompt's pre-answer reading state
with the original-text prompt's state in the same frozen body. The teacher
uses the original visible prompt only, before any correct answer history.
It is an offline training helper with the existing authenticated weights,
not another model download or an inference service.

## Fixed architecture and objective

Retain the complete native Show-o2 HQ source and FP32 Wan codec. The pooled
high-resolution zero-shot control also fails (0/32 Chinese, 0/32 English).
A replacement must preserve the source's native low-resolution reading at
initialization instead of substituting that failed pooled feature stream.

For each complete source image, compute its native whole-square 432 view and
its original 1024 view. Keep the low-view source frontend frozen. Encode the
high view with nine native semantic windows, pooling its fused visual features
to the native 729-position interface. Add a learned correction
G(high_features - low_features) to the low-view visual features. G is a
1536-by-1536 bias-free linear layer initialized exactly to zero. It adds
2,359,296 trainable parameters. The 158 fine-view semantic/fusion linear
layers use fresh gated rank-16/alpha-16 adapters (8,489,984 parameters).
Total new trainables are 10,849,280. All original source weights, the native
language body/head and image writer remain frozen. Never initialize from the
failed adapters. At zero correction, the reader is exactly the measured
native-432 source; the fine-view branch cannot change its initial answers.
Pooling the fine view is explicitly lossy, although all input pixels participate.

For the same visible source prompt, let t_l be the detached final-assistant-role
state of the original-text path at layers l in {4,8,12,16,20,24,28}. Let s_l
be the image path's corresponding pre-answer state. Normalize each vector
by sqrt(mean(h^2)+1e-6). Optimize mean squared normalized-state error across
these layers plus the existing native answer cross entropy, with coefficient
one for each loss. Correct answer history may enter the CE term but cannot
enter either pre-answer state. The source attention mask must prevent later
answer labels from affecting those states. The teacher has full original
text context and remains a privileged control, not proof of image ability.

The current image-only and text-input diagnostic scores set references for
comparison; no state or label from validation enters optimization. The
objective is representation distillation and supervised visual access, not a
claim of a novel foundation algorithm or vocabulary-free training.

## Training data, budget and execution

Reuse all original 384 training examples: 128 Chinese/English new-string copies
and 256 Chinese natural questions, with complete original square source pages.
Add 256 English questions from 128 distinct SQuAD paragraphs. Exclude every
article in the fixed 32-question English assessment. Choose complete pairs
with distinct answer spans by seeded source-hash order, at most eight pairs
per article. Preserve all source offsets/alternatives and reject only missing
font coverage or full source/reference page overflow. Use the existing
32-pixel source font and complete 1024-square pages. No answer crop, manual
correction, teacher outcome or validation result selects training records.
There are 640 examples, with an explicit inventory of exclusions and hashes.

Prepare detached training-only text states and sampled FP32 image latents
before optimization. Use fixed per-ID seeds and authenticate every cache item.
Run four complete independently shuffled passes over all 640 IDs, exactly
2,560 updates. AdamW uses betas (0.9,0.95), weight decay 0.01 and gradient
norm clipping at one. Warm up to 2e-5 over 64 updates, then cosine decay to
2e-6. There is no coefficient or learning-rate search on validation.
Use BF16 body/adapters, FP32 losses and the existing checkpointed frozen body.
Checkpoints retain adapters, optimizer/RNG state and source/data/code bindings;
resume only unchanged runs. Keep one GPU job active at a time.

Before the full run, execute three real updates on the largest target,
largest complete transcript and the first English question, starting again
from the source afterward. Require exact source prefix/mask/decision replay on the first four fixed
English native-source cases before optimization. Verify a zero-correction
source prefix match, every
adapter's finite gradients, changed residual tensors and an unchanged disabled
source path. Also verify that the training mask blocks every pre-answer-to-answer
attention edge and that pre-answer states agree under appended correct
history in a temporary FP32 control (normalized max error at most 2e-5). A failed preflight stops optimization.

## Complete autonomous assessment and claim boundary

Assess all 32 held-out copies at the same whole-square interface, all 48
Chinese paired conditions and all 80 English normal/erased conditions.
Measure the new source copying baseline before learning; reuse the exact
authenticated native-432 source Chinese/English observations for the remaining
128 calls, checking every derived low-view image hash and seed. After learning, decode from fresh images and numeric adapters,
without teacher states, transcripts, correct histories or training caches.
Keep all responses, including repetitions and cap failures.

A candidate for the later image-writing stage must reach at least 30/32 exact
stopped copies, 16/32 Chinese answers and four both-correct Chinese pairs,
and retain at least 11/32 English answers and three both-correct English pairs.
It must also improve Chinese exact answers over the frozen pooled source by
at least four. These are prerequisite gates, not completion of the ILM goal.
Do not claim language learning from state loss, copying, gradient checks or
memory alone. Compare exact answers and paired erasures separately from
legibility and descriptive character F1.

All current outputs still use an auxiliary Unicode head. A successful reader
must subsequently produce legible correct answers directly through a learned
continuous image path, without a teacher answer, font lookup or runtime text
decoder. No success in this protocol completes that independent image-input/
image-output requirement. All source counts, inherited pretraining and local
costs must remain explicit.

CMRC 2018 and SQuAD 1.1 retain CC-BY-SA-4.0 attribution to Cui et al. (2019)
and Rajpurkar et al. (2016). Both public datasets may occur in source
pretraining; the repeatedly inspected development cohorts are not sealed tests.

## Execution deviation and additional source check

The initial preflight actually replays four raw native-source readouts, and
compares zero-correction prefix/mask equality on its largest-target training
case. It does not execute the broader four-case corrected-path prefix/mask
and readout comparison described above. Main optimization has already started.
Preserve that timing deviation rather than describing the first preflight as
the complete four-case corrected-path check.

A separate fresh-source zero-correction check must execute all four corrected
prefix/mask comparisons and eight full readouts after the active GPU work is
terminal and before final quality qualification. No learned checkpoint or
quality outcome selects those cases. The final scorer must require that
receipt. This closes the missing source-equivalence observation without
pretending it occurred before optimization. The declared quality gates,
dataset, loss, optimizer and budget remain unchanged.
