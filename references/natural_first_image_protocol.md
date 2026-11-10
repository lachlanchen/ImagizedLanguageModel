# Natural first-image learning

The failed full-unit synthetic transfer is preserved. This experiment replaces
counterfactual vectors with states from actual complete question images. It
tests a first-image generalization repair, not a new language foundation.

Prepare all 10,142 questions in the pinned raw CMRC training partition. Exclude
every raw development title/context, validate answer spans, retain complete
paragraphs and questions without answer-aware cropping, and render three fixed
prompt styles. Use the existing complete-line Noto renderer and lossless
blank-column folding. Unknown font glyphs and rendering errors are explicit
rejections. Duplicate physical inputs must have consistent targets. Offline
Unicode is a data-preparation facility; the actor receives image pixels only.

For questions already in the inherited short-training partition, preserve its
first accepted reference to prevent contradictory targets. Otherwise use the
first validated source reference. The model predicts that answer's first
32-pixel canonical glyph. Preserve all allowed reference initials for evaluation.
Unsupported training initials are explicitly excluded from writer updates;
unsupported held cases remain failures in the denominator.

Hold out every fifth context under the fixed SHA256 namespace
`natural-first-held-v1`, but only if it never entered the inherited training
partition. Keep all questions/styles for a paragraph together. These held
contexts come from the dataset's raw training partition and are first-image
calibration holdouts, not a new official language benchmark. They do not overlap
the earlier reserved development cohorts. Inherited source pretraining exposure
is unknown. No outcome-dependent filtering or checkpoint selection is allowed.

Load the unchanged balanced parent (`dcac4ee2…`) and its fixed contextual
adapters. Cache the actual final state for each complete image, without output
history, teacher lexical projection, tokenizer or answer input. Retain immutable
shards and resume only matching source/image/checkpoint hashes. This cache is
valid because reader/body adapters remain frozen throughout the subsequent fit.

Prospective writer fit: initialize from that balanced parent and update G alone
for 36,000 batch-256 AdamW steps, learning rate 1e-4 cosine to 10%, weight decay
.01, gradient norm 1, seed 202611123. Use 128 supported natural first states,
64 inherited gold first states, 32 inherited later states and 32 genuine-copy
training states. Equal per-image foreground/background BCE has group weights
.5/.25/.125/.125. Retain E, all D, normalization, stop and body exactly. Natural
states are sampled uniformly over retained training images; all exclusions and
actual exposure counts are reported. The first-state cache does not train
stopping, complete answer syntax or generated-history behavior.

Before language rollout, compare parent/candidate in deployed FP32 arithmetic
on every held-context first state, all inherited training states and the
existing cached held-wording copies. Preserve every image and all supported
reference scores. Require held-first mean foreground F1 at least .90 and at
least 80% reaching .95, inherited training mean .98, cached-copy mean .95 and
80% of copies reaching .95. These are first-image qualification thresholds,
not answer accuracy. Complete autonomous language still requires a separately
registered third cohort, actual full-reader copying/stopping and unrepaired
output-first reading before reference scoring. Keep both prior reserved cohorts
as development evidence; never relabel them as a fresh test.

Preparation is CPU-only with 48GiB entry and 24GiB live reserve. Caching uses
the existing single-GPU0 mutex/full-reader admission (48GiB entry, 24GiB live,
22GiB allocated CUDA cap), at most 3,600s cumulative low-memory-pressure waiting
per 10,800s attempt. No second project heavy job or publication build overlaps.
Small-writer training/qualification use the same mutex, 48/32GiB host reserves,
14GiB CUDA cap, 3,600s stage cap and 1,800s pressure waiting. Preserve every
failed attempt and leave other projects' processes untouched.
