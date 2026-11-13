# Complete-corpus joint-pixel writer comparison

## Fixed change and controls

Replace only the final continuation model's marginal image writer with
`GlyphJointDensity`: four 192-wide layers, six heads, 128 physical 2-by-4
positions, 4,096-dimensional conditioning and 256 possible eight-pixel masks.
The final 41,298-update reader, residual dynamics, query, pixel-feedback encoder,
affine statistics and stop predictor remain frozen. Keep the old point writer's
complete final streams and scores as the unchanged control. Their authenticated
files are reused, not generated again or relabelled as new measurements.

The hypothesis is that within-image dependence avoids the measured dense pixel
mixtures while preserving conditional language information. A normalized image
distribution does not guarantee a correct form or a coherent answer. The earlier
512-record V68 physical head and 23,262-state flow both failed their language
pilots. This comparison instead uses the actual newly trained reader and every
eligible complete source; it does not revise those failures.

## Data and causal cache

Use the existing 13,766-row full-trial manifest, exclusions, group splits, target
cuts and original pixels: 9,004 QA and 4,762 book records, with 2,036,247 complete
target glyphs per layout. Cache both prompt layouts, giving 27,532 training
records and 4,072,494 target-predicting states. Do not add a teacher, tokenizer,
source decision head, OCR or new font rendering. For target t, cache the state
after the query and only preceding reference images. Store terminal states
separately in the same array; they never predict an image target.

States remain lossless FP32 and target images are losslessly packed into
eight-bit physical masks. Cache one sequence at a time, authenticate inputs and
outputs, and verify the inverse packing against every original target. The
fixed model and existing completed long-cache controls bind the cache; do not
repeat the completed diagnostic as another result.

Qualification uses the previous 124-case clean-history cohort. Its 48 training
jobs reuse the new training cache. Its 76 development jobs are cached separately
and excluded from all fitting and normalization. Preserve all 124 complete
targets and supplied-history labels. This is reused development, not an
untouched confirmation set.

Compute condition mean/variance only from training states, weighting each
sequence equally and fixed/proportional layouts 1:2, matching their schedule.
Clamp standard deviation below 1e-4. Stream reductions; do not load the complete
state corpus into RAM. Record source hashes, cache coverage and actual storage.

## Learning and checkpoints

Start the new head from seed 2026111339, archiving its exact initial parameters
before neural caching. Training-only normalization buffers are filled afterward.
Optimize the mean negative joint-image
log likelihood within each complete sequence, with each image's 128 mask losses
equally weighted. Process at most 64 glyphs per microbatch, accumulate the exact
complete-sequence reduction and make one AdamW update per source presentation.
All three epochs contain every source once, giving 41,298 retained updates and
6,108,741 target exposures. Each source uses its fixed layout once and its
proportional layout twice. No development loss selects a schedule or checkpoint.

Use FP32 weights/activations/moments, deterministic math attention, AdamW
betas (.9,.999), epsilon 1e-8, weight decay .01, peak learning rate 2e-4,
500-update linear warm-up, cosine decay to 2e-5 and norm clipping at one.
Use explicit log-softmax/gather for the categorical likelihood. Save initial,
10,324/20,649/30,973/41,298-update snapshots. Periodic resumable checkpoints
retain model, optimizer, random state, schedule position and ledger hash.

At actual step three save state, execute steps four/five, restore step three
and replay those same updates. Require identical losses, model/optimizer/RNG
state; the replay adds two executed updates but no retained exposure. Preserve
failed attempts and discarded work. No automatic retry or budget reset follows
a failure. A manual resume uses its last authenticated checkpoint and discloses
all repeated/discarded computation.

Check the head on actual cached training states at initialization, step five
and the final snapshot. Teacher-prefix parallel/cached conditional probability
vectors must differ by at most 1e-4 in total variation at every physical position.
Also require identical greedy physical choices when the parallel path sees the
cached path's own complete generated prefix. Report raw logit errors separately;
a shared additive logit offset need not change a distribution. Stop on failure,
without loosening this prospective rule. These checks do not prove language.

After fitting, an independent CPU audit follows the final event-hash chain back
to initialization, checks every retained source/layout/target exposure, verifies
lossless cached pixels against original target hashes, reaggregates training-only
normalization and opens every fixed snapshot's parameters and Adam counters.
It has a 900-second cumulative ceiling, two CPU threads, 8 GiB peak RSS and
24 GiB available-host reserve. Reader freezing remains a bound execution check;
this CPU audit does not repeat the neural cache. It must pass before qualification.

## Conditional writing review and language evaluation

Generate final-head glyphs autonomously within each image for the 124 fixed
correct-history cases, at the first position and up to eight evenly spaced
later positions. Keep the whole reference-conditioned physical metrics as
diagnostics. Do not require canonical next-glyph agreement to establish legible
book writing: a real language distribution may choose a valid alternative.

Mix candidate and unchanged-parent physical outputs under opaque IDs. Inspect
all selected tiles before unmasking their owner/target; freeze clear visible
readings, blank and uncertain labels. The prerequisite to another full answer
run is at least 90% confidently readable nonblank forms in every candidate
split/kind/layout group. Repeated readable forms can pass this physical gate
but must still answer the complete language tests. No target repairs a reading.
If this physical gate fails, stop this replacement before another unqualified
autonomous language run; unrun language scores remain unmeasured, not zero.

After that gate, evaluate the five fixed snapshots on the existing full core
suite: 216 endpoint and 76 interior streams, plus both endpoint 640-cap book
controls, totaling 708 new candidate streams. Greedy physical-mask decoding is
fixed; use one output per case and actual emitted-pixel feedback. No reference
length, lexical inventory, renderer, retrieval or teacher enters generation.
Review complete outputs before opening new mappings/references, retain every
stop/cap/uncertain reading and reuse the unchanged control scores explicitly.

The final pilot retains the full-trial rule: proportional exact >=12/32,
both-correct >=4/16, readable >=24/32 and coherent complete long books >=4/12;
retain all eight required proportional and both fixed exact cases. Every one
of four QA-erasure arms must be <=2 exact; the proportional normal-minus-better-
erased gap must be >=8. An intermediate checkpoint cannot replace the final.
An apparent pass requires untouched source-group confirmation and standalone
execution/resource checks before usability, generalization or efficiency claims.
Clean forms without improved complete language reject a writer-only account.

## Resources and artifacts

Use physical GPU0 under the existing project mutex, two CPU threads, no GUI and
no concurrent heavy job. Preserve GPU1 and other projects. Cache: 48 GiB free
host RAM and 20 GiB free VRAM at entry; 24 GiB host reserve and 22 GiB peak
allocated VRAM at runtime; 24-hour cumulative stage ceiling with at most two
hours of memory-pressure waiting. Head fit: 24 GiB host entry/reserve, 20 GiB
free VRAM at entry, 8 GiB allocated ceiling, 12-hour cumulative ceiling and
one-hour pressure allowance. Qualification: 8 GiB allocated, one hour total,
300 seconds pressure waiting. Future full generation: 22 GiB and eight hours.

Require at least 120 GiB free disk before cache creation and 48 GiB afterward;
stop before exceeding 100 GiB of generated cache data. Lossless cached states
are expected to occupy about 63 GiB plus small targets/metadata. Do not duplicate
shared model weights. Save private arrays/checkpoints/ledgers in `artifacts/`;
commit only code, protocols and non-sensitive summary bindings. All stage times,
including failures, remain separate from inherited pretraining and previous fits.
The deployed whole body remains large; 2,642,368 trainable head parameters do not
establish a compact independent foundation or cheaper language learning.
