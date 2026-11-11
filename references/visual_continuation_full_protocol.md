# Complete-sequence continuation adaptation: fixed full trial

The 18.123-GiB real preflight passes, including full 513-glyph targets and all
36 answer-layer gradients. The present trial tests complete language behavior
with substantially broader source exposure. Preflight updates are discarded.
The current strict proportional baseline remains 6/32 balanced and 7/32 original
flow, with no qualified book continuation; neither is a usable ILM.

## Model, data and exposure

Use the unchanged continuation-only architecture: frozen source prompt prefill,
one learned continuous query, rank-eight/alpha-sixteen residuals after all 36
native layers, the existing image readout and learned stop head. Source vision,
context weights, original zero-output source adapters, E/D and affine statistics
stay frozen. Initialize the image bridge from the bound balanced checkpoint,
without importing the trained four-layer source adapters. Start fresh residuals
and query with seed 2026111231. No preflight checkpoint or held-out result enters
initialization. This remains transfer of inherited 7.522B-body language weights.

Use every eligible record of the authenticated 13,785-source-sequence training
split. Preserve source/group separation and exclude the earlier feedback-ceiling
helper controls and their groups, plus current held-out source groups. Do not
filter targets by the 4,087 primitive-image inventory. Every QA answer is complete.
For each book record with at least two glyphs, expose its first
min(128,max(1,floor(total/4))) glyphs and predict the entire remaining record.
Record actual target counts after this prefix split and exclusions. The original
2,576,239 target count is not the new exposure count.

Prepare both original 32-square fixed-cell pages and natural proportional pages.
The latter use the unchanged Pango CompleteLineRenderer and lossless blank-column
folding from the prior input-presentation study. Replace invisible controls and
whitespace with spaces without changing visible source content. Do not crop,
shrink or remove visible text. Font/layout rejections fall back to the complete
original fixed raster and are counted; at least 95% must render naturally.
All rendering is offline. Fonts, text strings, source IDs and lexical tables do
not supply deployed answers.

Three epochs cover every eligible sequence once each. Deterministic CPU
permutations set order; each source gets two proportional and one fixed
presentation across the three epochs. Bind manifest, source shards, images,
schedule, source/runtime versions and exclusion records before training.

## Optimization, snapshots and recovery

Batch one complete sequence per update, using the preflight's balanced physical
reconstruction loss on all glyph targets plus 0.1 times balanced stop/continue
BCE. The one terminal boundary is included. Use FP32 AdamW parameters/moments,
no weight decay, peak learning rate 0.0001, linear warmup 500 updates, cosine
decay to one tenth of peak and global gradient clipping at norm one. No source
vocabulary projection or additional teacher call participates in this fit.

Retain initial, quarter, half, three-quarter and final snapshots at the integer
positions fixed by the complete schedule. No checkpoint/seed selection or early
stopping from development answers. Complete the predetermined recipe unless a
nonfinite value, identity failure or registered resource limit stops execution.
Intermediate readable examples do not change it. A failed final pilot rejects
this bounded recipe; do not extend an unchanged fit.

Save initial trainable tensors, full optimizer, Torch/NumPy/Python RNG states
and exact schedule position. Use immutable checkpoint files and an atomic pointer;
retain fixed snapshots, current/previous optimizer state and actual resume-check
evidence. Remove only generated superseded optimizer files. Preserve all ledgers
and interrupted attempts. Checkpoint every 250 updates and at fixed snapshots.

During the first six scheduled updates, save step four, continue through six,
restore four and replay five/six. Require bit-exact trainables, optimizer and RNG
against the actual uninterrupted control. These two verification updates are
extra compute, not added to nominal source exposure. Preserve their losses and
checkpoint hashes. On restart, verify PID/start/boot identity and use the saved
checkpoint and data order. Never start a second worker because observation timed
out. Source/body/interface versions must remain unchanged during fitting.

## Complete-output evaluation and review

After fitting, evaluate all five fixed snapshots from their saved weights.
Use the existing 216 physical inputs, unchanged erasures, threshold 0.5 and
128-glyph cap at initial/final; evaluate the 76 normal inputs at all three
interior snapshots. These are 660 core streams, including both QA layouts and
book prefixes. Feed only actual emitted binary pixels as continuation history.
Supply no reference answer, target length, language ID, font, OCR or external
teacher. Keep initial weights as an actual checkpoint, even if evaluated later.

Separately generate all twelve normal and twelve erased book prefixes at cap
640 for initial/final candidate and the frozen balanced parent: 72 additional
streams. The parent uses its exact existing body/bridge checkpoint in a separate
serial process. These new matched-cap arms do not replace or rescore the older
128-cap results. They test continuation lengths compatible with full book targets.

Prepare every raw image and opaque review board before new reference scoring.
Freeze complete transcriptions/uncertainty and book-coherence judgments. The book
judge sees the corresponding visible prefix. A coherent continuation must add
readable, syntactically coherent material consistent with the prefix, rather than
only repeat it, restate instructions or drift into malformed/repetitive glyphs.
Uncertain streams are not repaired from references. Apply the original strict
QA scorer unchanged to all core snapshots. Report complete-answer accuracy,
both-correct pairs, readability, natural stops/caps, erasures and exact-case
retention. Canonical mismatch alone does not establish semantic error.

The final development pilot requires at least 12/32 strict exact proportional
answers, 4/16 both-correct pairs, 24/32 readable proportional answers and 4/12
coherent complete long-book continuations. Retain all eight union proportional
and both original fixed exact answers. Each QA-erasure arm may score at most
two exact answers; normal minus the better erased arm must be at least eight.
Report the matched-parent book results separately. A pilot pass needs untouched
source-group confirmation before any claim of usability or generalization.
Prior cohort familiarity and absence of independent human adjudication remain
explicit. No semantic interpretation or image reconstruction metric can rescue
a failed registered strict pilot after results are known.

## Resource and reporting limits

One heavy GPU0 job under the existing mutex, in a single named project tmux
session. Controller children are serial. Source weights and environments are
reused. Entry requires 48 GiB available host RAM and 20 GiB free GPU memory;
runtime requires 24 GiB host reserve and at most 22 GiB allocated VRAM. Poll
host pressure every half second with bounded sample storage and check GPU/time
ceilings through native-layer hooks. Preserve foreign services, including GPU1.

Training has a cumulative 48-hour wall budget; each attempt permits at most
two hours of pressure waiting within that wall budget. Candidate snapshot
evaluation and parent evaluation each permit six hours, including at most
thirty minutes pressure waiting per attempt. Include loading, interruptions,
replays and data preparation in cost reporting. Peak memory is not a quality or
efficiency result. Record total unique deployed parameters as well as trainable
parameters and inherited pretraining. The full independent visual-language and
historical/unencoded objective remains active until its actual criteria hold.
