# Test whitespace learning without discarding the nonempty-glyph gain

The [ranked writer](global_glyph_ranking_physical_result.md) improves conditional
readability in every required group and reduces false blanks from 711 to 37,
but fails legitimate whitespace with 25 false-ink outputs against a maximum of
eleven. The [completed density diagnostic](ranked_writer_whitespace_result.md)
rejects a fixed blank-versus-emitted-image decoding rule: only three spaces
would be fixed and one additional nonempty form erased. Preserve both results,
all original arrays, frozen readings, final weights and failed gates.

## One objective change

Keep the existing normalized physical-mask density, supplied-history training,
greedy writer, native prompt/body, feedback, stop head and complete source
schedule. Retain the current nonempty source-versus-blank ranking term. Add a
term **only at truly empty training targets**; derive emptiness from source
pixels rather than Unicode labels or an inferred space class.

Let `S = log q(X|h)`, `B = log q(Z|h)` for complete source image `X` and blank
image `Z`, and let `U` be the existing first-nonzero-prefix upper bound on every
nonempty image's log probability. For `a(X) = 1` when source pixels contain ink,
the proposed per-position loss is

```
L(X,h) = -S/128
       + a(X) softplus(1 + B - S)/128
       + (1 - a(X)) softplus(1 + U - B)/128.
```

Average over the complete source's target positions and retain 0.1 times the
unchanged balanced stop loss. Both ranking strengths and complete-image margins
are one. The two terms act on disjoint target positions. The blank-path pass
already supplies the zero-prefix probabilities needed for `U`; no glyph
catalogue, lexical classifier, live teacher or sampled hard negative is added.

For normalized probabilities, every nonempty image first differs from the blank
at some mask. Its remaining suffix has log probability at most zero, so its
complete log probability is at most `U`. Thus `B > U` is sufficient for blank
global optimality in exact arithmetic. Reducing the proposed softplus penalty
encourages that inequality at white targets; it does not guarantee achieving it
or learning correct language. The bound can be loose. Its maximum supplies a
subgradient at ties, and strong blank pressure can erase nonempty writing.
Those are explicit ways this intervention may fail.

## Proof required before another full fit

Implement the change in a new module while leaving the 101 bound diagnostic
sources unchanged. Compare all context/head gradients with unsplit autograd on
mixed, all-white, all-nonempty and non-divisible CPU batches; include zero added
strength to reproduce the existing ranking gradient. Verify the first-nonzero
bound independently on an enumerable small physical alphabet, including ties.
These are implementation checks, not language measurements.

Before GPU execution, bind code, runtime, initial weights, all source images and
targets, selection rule, limits and a fixed acceptance decision. Use the two
original complete five-glyph QA layouts for actual 131-tensor unsplit gradient
comparisons and both complete 513-glyph book layouts for memory. Additionally
use both layouts of the **first original qualification training-book source
containing both white and nonempty targets**, in the unchanged qualification
order. Bind its actual identity and complete target arrays before execution.
Do not crop a convenient span or synthesize a blank into the source. Take real
complete-source optimizer steps for the four book cases, verify finite gradients
and exact counters for all 131 tensors, preserve frozen body/feedback/stop and
affine buffers, and reset every trial update to the identical fresh initialization.

One GPU0 owner under the project mutex, two CPU threads, 1,200 seconds including
180 seconds of pressure waiting; 48 GiB available host RAM and 20 GiB free GPU
at admission; retain 24 GiB host RAM, 48 GiB disk and at most 22 GiB allocated
VRAM. External TERM at 1,180 seconds and twenty seconds of kill grace bound
setup as well as forwards. Preserve first attempts and partial costs; no
automatic retry or duplicate worker. Do not alter foreign processes or install
another copy of shared weights.

## Fixed outcome and later evaluation

Only a passed, reset preflight permits registering a full matched comparison.
Initialize from the same fresh head and point-final context as the completed
ranking control, not an intermediate or fine-tuned final checkpoint. Keep all
13,766 sources, 41,298 updates, 6,108,741 exposures, five snapshots, optimizer
schedule and 131 trainables. Reuse unchanged ranked control pixels.

Retain all 124 conditional cases, actual opaque review before its first freeze,
all seven 90% readable-group requirements, independently verified credited ink
and at most eleven false-ink outputs among 252 true-white positions. Add a
prospective safeguard that all-case false blanks must not exceed the current
control's 37. No outcome changes an earlier rule or score. A failure stops the
candidate before autonomous evaluation and redirects learning or representation
from the actual new error, rather than extending epochs or tuning thresholds.

Only an actual new physical pass permits a separately committed complete
708-stream language comparison. Paired prompts, erasures, 128/640 caps, natural
stopping, earlier-success retention and the original pilot remain unchanged.
Untouched confirmation, independent execution, historical forms and whole-model
matched-quality efficiency remain required. This decision launches no fit and
claims no language success, compact foundation or novel physical theory.
