# Fixed complete-image learning on the varied Chinese cohort

This experiment tests whether the working physical writer generalizes when
conditional supervision spans 512 questions in 256 training articles. The
preceding 32-question run fits every complete answer but gives no usable normal
development answer. The next run preserves its native image actor, normalized
physical density, full answer-image loss and original initial state. It does
not assume that more data repairs the failed coupling.

Use the prepared and bitstream-audited `artifacts/pixel_language_diversity_v1`
cohort: 32 original questions plus 480 questions from 240 additional articles.
There are 1,657 complete answer blocks and 498 distinct complete answer images.
Answers have one to seven 32-by-64 blocks; the largest input has 393,216 pixels.
No generated quality or teacher performance selects these examples. Preserve
the exact 24 reused development questions, their normal/erased PNGs and all
complete reference alternatives. English remains untrained transfer here.
Metadata identities remain outside model inputs. The actor receives pixels.

Restore the original pre-qualification actor state, SHA-256
`e3949fc68937affb19e0ec22386b577f702dfb9e45b07f10c6d6cca735a160d6`.
Do not warm-start from the 32-question fitted model. Cold reconstruction uses
the actor's own state plus authenticated shared frozen source weights, history
encoder and numeric controls, without legacy training checkpoints or runtime
teachers. Retain the original external pretraining and prior raster learning
in the provenance; this is not a foundation trained from scratch.

The complete objective is unchanged: mean physical-piece NLL within blocks,
mean blocks within each question, mean questions in a batch, plus per-question
balanced continue/end softplus at equal weight. Every complete target piece
is supervised. No current or future image block conditions its own query.
The pixel head retains FP64 arithmetic and FP32 stored parameters. The source
body stays frozen and suffix adapters remain inactive during prompt prefill.

Use batch four, AdamW (.9,.95), decay .01 and global clipping at one. Physical
head and other peak rates are 3e-4 and 1e-4, with 100-update warmup and cosine
decay over exactly 16,000 updates, ending at one tenth of peak. This changes
both diversity and the schedule/exposure relative to the old run; it is not
an isolated data-only or matched-compute causal comparison.

Keep at most 32 complete native prompt KV records in a CPU pool. Shuffle all
512 training indices with a serialized private generator, partition the order
into 16 windows, and train four fresh local permutations of each window before
advancing. Each window gives 32 optimizer updates; a complete outer traversal
gives 512 updates and four exposures per question. The full 16,000-update
budget retains 64,000 example draws, averaging 125 exposures per question. The final
partial traversal is reported without claiming a completed outer cycle. This
amortizes prompt prefill while retaining all owning prompt keys. Pool/window,
local permutation, cursors, generator and optimizer state are checkpointed.
Discard obsolete pool tensors before loading the next window. Save no full
corpus KV archive and no copy of the frozen foundation weights.

Before sustained learning, run an actual GPU qualification from the declared
initial state. Test a variable-length batch with lengths 1/3/4/7 against
independent owning calls, using normwise state and relative objective tolerance
1e-4. Require exact future-image and padded-future-code exclusion. Also run
actual backward passes on the four largest prompt pages and the four longest
answers, with finite, nonzero required gradient groups. These actual stress
batches are admission checks, not a proof of a universal memory bound.
Perform two real scheduled updates, restore step one and repeat step two;
require bit-exact model, Adam, sampler, losses and gradient measurements.
The extra replay is counted. Discard all qualification updates before the
main run. The main run's first two updates must match that qualification's
model/Adam/sampler state and measurements exactly after a fresh cold load.

Generate source-fixed training indices 0--63 and all 24 normal plus 24 erased
development streams initially and at updates 4,000 and 8,000. At the fixed final
update 16,000 generate every one of the 512 training answers and all development
streams. The first 32 training cases are old controls and the next 32 are new
source-order examples. Do not select checkpoints by development performance
or stop on training fit. At the same four states, measure the whole 512-question
correct-history objective without optimization between its cases. Label it
separately from autonomous generation. All generation invokes fresh original
PNG inputs, re-encodes actual emitted images and uses learned stop threshold .5
with cap128, counting a cap as failure. Neither targets nor desired length enter
generation. Retain every original output, stop trace and actual feedback code.

Record initial full prefix hashes on first capture for all 512 training inputs.
Guard frozen parameter identities/versions throughout optimization. Every
pool refill invokes the real PNG reader; it is not a persistent learned cache.
At the end compare fresh complete prefix hashes for the original 32 controls
against their first capture. All final answer generations invoke the reader
again. These checks do not independently implement the inherited reader.

The sole project GPU0 process uses the existing lock, at least 24 GiB host RAM
available, at most 75% swap use on admission and a 22 GiB allocated-memory cap.
Use a 12-hour active-process deadline, counting earlier active segments during
an explicit resume. Preserve GPU1 and foreign services. No Office, LaTeX or
second project model job overlaps. Keep current/previous rolling checkpoints
every 250 updates. Resume is explicit only after the original process is absent;
an observation timeout is never a restart instruction. Preserve uncommitted
history tails and partial outputs. Actual executed optimizer calls, including
discarded work on resume, remain in a durable separate ledger.

After process exit, independently score complete images against every owning
reference, stop/cap events, paired questions and erasure effects, and re-encode
actual feedback images on CPU under the existing 1e-4 tolerance. Inspect every
normal development image for readable, correct content. Pixel overlap alone
is not semantic accuracy. Report complete training and reused-development
results regardless of quality. This experiment cannot establish historical
scripts, broad instruction following, a lightweight foundation or efficiency
at matched LLM quality. If unseen answers remain unusable, record that failure
and investigate the coupling rather than repeating unchanged fitting.
