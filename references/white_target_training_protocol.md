# Complete matched white-target ranking comparison

The [completed ranked actor](global_glyph_ranking_physical_result.md) passes all
seven conditional-readability groups and reduces false blanks to 37, but fails
true whitespace with 25 false-ink outputs against a maximum of 11. The
[complete density diagnostic](ranked_writer_whitespace_result.md) rejects the
fixed blank-image comparison. The [new reset preflight](white_target_preflight_result.md)
passes all six actual complete-source cases, including both 314-glyph mixed-white
books and all 131 QA gradient comparisons. This protocol tests the single
[selected objective change](white_target_ranking_decision.md), not more epochs
of the failed recipe or a new decoding rule.

## Fixed learning comparison

Initialize the identical original fresh physical head, point-final context and
training-only normalization used by the completed nonempty-ranking actor.
Preserve the same 131 tensors and 5,005,760 trainables. Native prompt/body,
emitted-image feedback encoder, stop parameters and four affine buffers remain
frozen. The whole deployed model is 7,553,448,370 parameters, including
7,522,463,984 inherited parameters; small trainable count is not compactness.

For each target image `X`, let `S=log q(X|h)`, `B=log q(blank|h)` and `a(X)`
indicate visible source ink. The normalized physical density is unchanged. Add
the white-target term to the existing complete-source objective:

```
L = mean_t [ -S_t/128
           + a(X_t) softplus(1+B_t-S_t)/128
           + (1-a(X_t)) softplus(1+U_t-B_t)/128 ] + 0.1 L_stop.
```

`U` is the first-nonzero-prefix upper bound on every nonempty image. White
source-path logits already supply its blank-prefix scores. Both strengths and
whole-image margins are one. The old nonempty term is retained; the new term
acts only at true-white source pixels. The combined loss is not pure maximum
likelihood or a calibration guarantee. The unchanged greedy writer still emits
physical images without a Unicode catalogue, occupancy override or forced pixel.
The predicted benefit is improved legitimate whitespace without losing readable
nonempty forms. Excess blank mass, poor glyphs or language failure can reject it.

Keep all 13,766 complete sources, 41,298 retained updates and 6,108,741 target
exposures: one fixed-cell and two proportional presentations per source, with
the exact original ordering. Recompute actual prompt memory and causal context
for every complete source; no predictor cache replaces the trained state.
Microbatch 64 glyphs, accumulate their exact state gradient and differentiate the
actual continuation graph once. Keep FP32 AdamW, peak 2e-4, warm-up 500, cosine
floor 2e-5, weight decay 0.01, betas (0.9,0.999), epsilon 1e-8 and clipping one.
Save steps 0, 10,324, 20,649, 30,973, 41,298 and resumable current/previous checkpoints
every 250 retained updates. No preflight optimizer state or update is carried.

Actually restore step 3 and replay 4/5, requiring exact weights, Adam state, RNG,
losses and gradients. Count the two extra discarded updates separately. Every
ledger event records both ranking terms and the white/nonempty target partition.
The independent CPU audit authenticates all original source pixels, the complete
schedule/exposure, event chain, all 131 Adam states at every fixed snapshot and
the saved replay. It recounts 68,808 white-target exposures from actual complete
source images. Frozen-state assertions are bound execution checks rather than
an independent second neural replay.

## One serial owner and unchanged review

Commit implementation and contract checks, then CPU-register exact sources,
data, runtime, initialization, controls and limits. Commit and push that
registration before GPU work. One controller runs fit, independent CPU audit
and all 124 conditional qualification cases serially, then exits before review.
No automatic retry, duplicate fit or autonomous generation. Preserve failures,
partial outputs and cumulative prior costs before a justified manual recovery.

Fit has 108,000 seconds including 7,200 pressure-wait seconds; qualification
3,600/300 seconds; CPU audit 900 seconds at 8 GiB peak RSS. GPU0 mutex only, two
CPU threads; admit at 48 GiB available host RAM and 20 GiB free GPU, retain 24 GiB
host RAM and 48 GiB disk, and at most 22 GiB Torch-allocated VRAM. Reuse verified
shared weights and the existing process-local NVIDIA wrapper. Protect foreign
projects. Do not launch another heavy project build while this owner runs.

Qualification uses actual new context states and supplied source-image history.
Reuse the completed ranked actor's saved candidate pixels as the unchanged
control. Preserve all 124 cases, 12,597 positions, selected positions and all
original source arrays. Inspect every opaque board, derive blank flags from
original pixels, retain uncertainty, and freeze once before new owner/reference
scoring. Never repair old frozen transcriptions or regenerate control images.

Require all seven original candidate groups at least 90% readable nonblank,
independently credited ink, at most 11 false-ink outputs among exactly 252
source-white positions, and at most 37 false blanks across all 12,597 positions.
The final additional safeguard is prospective; earlier gates remain unchanged.
Canonical bitmap identity, glyph readability, whitespace and language are
separate measures. New model-based review must disclose prior familiarity and
the absence of independent human adjudication.

A physical failure stops this actor before autonomous evaluation and redirects
from the measured new error; it does not authorize threshold tuning, selecting
an intermediate snapshot, extra epochs or smaller data. A pass permits a
separately committed execution binding for the unchanged 708-stream comparison,
restoring all 131 actual learned tensors at each fixed snapshot. Keep emitted
pixel feedback, stop threshold 0.5, caps 128/640, paired questions, all QA erasures,
prefix erasure, earlier-success retention and complete long books. The pilot
remains at least 12/32 strict proportional answers, 4/16 both-correct pairs,
24/32 readable answers, 4/12 coherent long books, all eight proportional/two
fixed earlier successes, each QA-erasure arm at most two exact answers and a
dependence gap of eight. Untouched confirmation, independent execution,
historical forms and whole-model matched-quality cost remain full-goal criteria.

The Nature-style manuscript and editable native figures must distinguish the
completed failed comparisons, this fit's actual status and future complete
language. No usable language, new physical theory, compact foundation or
large-model parity claim follows from code, gradients or this registration.
