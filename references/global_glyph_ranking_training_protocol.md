# Complete whole-glyph ranking comparison

The completed [learned-context probability study](joint_context_blank_modes_result.md)
finds 108 certified empty global modes among false blanks, despite lower source
likelihood loss. Its search-only eligibility rule fails. This comparison tests
the single loss change in the [ranking decision](global_glyph_ranking_decision.md).
The [actual full-body preflight](global_glyph_ranking_preflight_result.md) passes
all four cases and all 131 gradient checks; every preflight update was reset.
Neither result establishes autonomous language.

## Fixed model, data and objective

Initialize the identical fresh physical head and original point-final context
used by the completed proper-likelihood comparison. The original writer supplies
the fresh-head draw and training-only normalization. The completed context actor
supplies the unchanged control output pixels. These are distinct sources.
Train the same 131 tensors, 5,005,760 parameters; freeze native prompt/body,
feedback, stop parameters and four affine buffers. The deployed model remains
7,553,448,370 parameters, including 7,522,463,984 inherited parameters. Small
trainable count does not establish a compact foundation or lower training cost.

For each complete source, retain mean physical-mask negative log likelihood and
0.1 times balanced stop loss. For each nonempty target add
`softplus(1 + log q(blank|h) - log q(source|h))/128`, with strength one, averaged
over all target positions. True whitespace receives its unchanged likelihood
and zero ranking penalty. The normalized 128-mask density and greedy decoder
remain unchanged. This combined loss is not pure maximum likelihood. The
prediction is fewer false blanks without damaging legitimate whitespace or
conditional readable forms; incorrect nonempty output can reject the change.

Keep all 13,766 complete sources, 41,298 updates and 6,108,741 target exposures:
one fixed-cell and two proportional presentations per source. Recompute actual
prompt memory and causal target-preceding context each update. Microbatch 64
complete glyphs through an accumulated state gradient, then differentiate the
context graph once. No cached predictor state replaces the actual graph.
Use the unchanged FP32 AdamW schedule: peak 2e-4, 500-step warm-up, cosine floor
2e-5, weight decay .01, betas (.9,.999), epsilon 1e-8, gradient clipping one.
Save fixed steps 0, 10,324, 20,649, 30,973 and 41,298, and current/previous
resumable checkpoints every 250 retained updates.

Actually restore step three and replay four/five; require bit-exact trainables,
optimizer, RNG, losses and gradients against uninterrupted execution. Count two
extra nonretained updates separately. The independent CPU audit authenticates
the event hash chain, every original source image/target, scheduled exposure,
all 131 actual Adam states at every fixed checkpoint and both saved replay
outcomes. It recounts the loss components and nonempty-target mask from source
pixels. Frozen-state assertions are bound execution checks, not an independent
second neural replay.

## Serial execution and review

Commit implementation, then CPU-register exact source/input/weight/runtime and
resource hashes, then commit and push the registration before GPU execution.
One durable controller runs fit, CPU audit and 124-case conditional qualification
serially and exits. No automatic retry, parallel fit or autonomous generation.
Preserve failures and partial outputs before separately justified recovery;
prior attempts consume cumulative limits. Fit: 108,000 seconds, including at
most 7,200 pressure-wait seconds. Qualification: 3,600/300 seconds. CPU audit:
900 seconds, 8 GiB peak RSS. GPU0 mutex only, two CPU threads, 48 GiB available
host RAM and 20 GiB free GPU at admission; retain 24 GiB host, 48 GiB disk and
at most 22 GiB Torch-allocated VRAM. Reuse verified shared weights and the
existing process-local NVIDIA compatibility wrapper. Do not alter other projects.

Use all original 124 complete conditional cases and selected positions. Actual
new context states use supplied reference-image history. Saved proper-likelihood
candidate pixels become this comparison's control; do not regenerate them.
Keep original source/candidate/control arrays and the seven groups. The same
90% readable nonblank threshold applies independently to every candidate group.
Canonical bitmap agreement remains a separate diagnostic.

The new review workflow derives empty/nonempty flags from original opaque
binary 32-by-32 pixels before the first freeze. It rejects contradictory flags
at recording, freeze and scoring without consulting owners or sources. All
boards must still actually be viewed; nonempty transcriptions and uncertainty
remain the reviewer's responsibility. Freeze once before owner unmasking. This
prospective safeguard does not repair any old readings or verdicts.

After freezing, verify every credited candidate tile contains ink and recount
all complete source-whitespace positions. The candidate may produce at most
11 false-ink outputs among exactly 252 source-white positions, the completed
control's count. Require all seven readable groups, the ink check and this new
whitespace condition to pass. Fewer blanks or loss alone does not qualify it.

Failure stops this actor before autonomous evaluation. A pass permits separately
binding the unchanged 708-stream comparison, restoring all 131 actual tensors
at each fixed snapshot. Preserve emitted-pixel feedback, frozen stop at .5,
128/640 caps, paired questions, all four QA erasures, prefix erasure, retention
and complete long books. The pilot remains at least 12/32 strict proportional
answers, 4/16 both-correct pairs, 24/32 readable answers, 4/12 coherent complete
long books, all eight proportional/two fixed earlier successes, each QA-erasure
arm at most two exact answers and a dependence gap of eight. A pilot pass still
requires untouched confirmation, independent execution, attested historical
forms and whole-model matched-quality cost. If writing passes but language
fails, redirect conditional language learning rather than claim typography as
understanding. Keep the full engineering goal active.

The Nature-style manuscript and native figures must distinguish the completed
failures, this trial's measured status and future language tests. No language,
novel-physics, lightweight-foundation or large-model parity claim follows from
the preflight or this protocol.
