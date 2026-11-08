# First complete full-page QA learning experiment

Register before the first QA update. This is a bounded fit/generalization test
of the existing continuous image writer, not a claimed new foundation model.
The independent ILM goal and the existing reliable-writing requirement remain.

## Question and fixed intervention

Can the writer learn to produce an answer to a question supplied only as pixels,
instead of copying its passage? Earlier copying experiments do not answer this.
The full-page probe established execution, not learning; its two discarded
updates are not used here.

Start from the complete ink-mass checkpoint with SHA256
`166409db4f87af87393576195251527f1fcf54d5fe12812d75f6d1491674717d`.
This checkpoint was selected after its copying comparison, before any QA
training or QA evaluation of these weights. It retains the pretrained 3.900B
transformer and 84.046M VAE, with 24.359M rank-16 adapter parameters. Source
language/image pretraining is inherited and must be counted. The earlier
four-question preflight used a different checkpoint and is not this baseline.

Use the first 32 training questions (16 complete passage pairs) in the existing
`pages_v2` manifest, with no legibility- or answer-based selection. Execute
four complete independent permutations: 128 updates. The complete 1024-square
passage/question page enters the native image branch. The answer is a complete
512-by-256 image. The fixed generic conditioning tensor is unchanged. No
varying text embedding, OCR transcript, target crop or answer enters sampling.

Fresh AdamW states, rate 1e-5, decay .01 and norm clipping at one are fixed.
Model seed is 20268901. Python `random.Random` order/noise streams are
20268902/20268903. Each target is loaded only after independent-noise generation.
Differentiate all 50 native sampling steps and the frozen decoder. The objective
is the already measured recognition CE plus 0.4035679875746343 times fixed
target-mass Bernoulli KL, temperature .05. No further loss tuning or mixed
copying rehearsal is used. Reader weights and vocabulary are training-only;
inference uses neither. The training-only coefficient comes from the prior
copying calibration, not QA validation. This does not guarantee semantic reading.

Before the first update, compare the actual differentiable output with native
inference at the same input/seed; require exact pixels and finite nonzero
early/middle/final gradients. Save each actual pre-update page and checkpoint.
Resume only with identical sources, data, schedule, weights, optimizer and RNG;
the checkpoint's embedded completed history is authoritative. Preserve partial
attempts rather than overwriting their pixels. Stop on nonfinite gradients,
source mutation, memory over 22 GiB or failed identity checks. Do not select a
checkpoint based on QA scores. Only final update 128 is evaluated.

## Complete assessment

Use the previously prepared 32 seen questions and all 32 article-disjoint
held-out questions. Each unseen passage has two questions with disjoint
accepted-answer sets. Evaluate both the final model and the fixed initial
model with identical input pixels, seeds, generic condition and native sampler.
Also evaluate all 32 held-out question-erased and 32 passage-erased controls
for both models. The question-erased pair has identical pixels and seeds;
the answer-label-position normalization is the disclosed layout intervention.
All images and answers remain complete, unrepaired and unselected. Targets and
training manifests are absent from the input-only inference process.

Training runs first. The fixed initial checkpoint is assessed afterwards to
avoid delaying optimization; no assessment changes the registered schedule.
This order is not an adaptive checkpoint choice. Assess final copying on the
unchanged full 64-case strip index; reuse the complete authenticated initial
ink-mass copying report, with its exact checkpoint and index bindings. Total
new native assessment calls are 320: 128 QA/control calls per checkpoint and
64 final copying calls. The initial copying report is not a new neural replay.

Run CPU OCR separately. Retain strict canonical exact matching and separately
score against every source-accepted answer with the same NFC/whitespace
normalization, retaining case and punctuation. Report complete paired-answer
counts and question/passage-erasure effects. Reference OCR is imperfect
(23/32 seen and 26/32 held-out canonical images). Raw inspection of every
normal/control answer remains required; OCR scores alone cannot certify writing.

Prospective exploratory requirements, all required:

- At least 24/32 correct seen answers by source-accepted OCR matching.
- At least 12/32 held-out answers and four more than the initial model.
- At least 4/16 both-correct held-out passage pairs and two more than initial.
- Held-out normal exact count at least four above each erased-input control.
- Final held-out copying no more than two below its fixed initial 19/32.

Passing these would establish only a bounded developmental learning signal,
subject to raw review, on reused public data and one schedule. It would not
establish reliable writing, book-scale language modelling, historical-script
understanding, efficiency superiority or the complete ILM objective. Failing
fit or question dependence is evidence against scaling this unchanged route.

## Execution

`scripts/run_native_paired_qa.py --stage prepare` binds every selected input,
source answer, source file and schedule on CPU. Commit the exact implementation
and protocol, then run `--stage run` through the established model Python runtime.
One GPU0 child at a time uses the shared lock; GPU1 and other projects are
preserved. No heavy publication build may overlap a model stage. Runtime
handoff, actual costs and errors remain under the ignored experiment directory.
