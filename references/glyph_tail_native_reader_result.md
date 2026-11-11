# Matched native decoding gives 10/32 exact answers versus 6/32 glyph answers

The complete helper diagnosis authenticates **448 calls** and reproduces all
**192 adapted image prefills bit-for-bit** from actual glyph generation.
With proportional input, the adapted native decoder gives **10/32 strict
complete exact answers**, versus **6/32** through the glyph writer. This is
a diagnostic contrast using an inherited lexical decoder outside the ILM;
it creates no new ILM images and makes no optimizer updates.

| Route | Fixed-cell image | Proportional image | Complete text prompt |
| --- | ---: | ---: | ---: |
| Adapted native exact QA | 4/32 | 10/32 | 11/32 |
| Native exact QA, adapters disabled | 2/32 | 13/32 | 10/32 |
| Adapted native both-correct pairs | 1/16 | 1/16 | 2/16 |
| Native both-correct pairs, adapters disabled | 1/16 | 2/16 | 2/16 |
| Previously generated adapted glyph exact QA | 2/32 | 6/32 | Not evaluated |
| Previously generated adapted glyph both-correct pairs | 1/16 | 0/16 | Not evaluated |

All eight erased-input image arms give **0/32 exact answers** after either question
or passage erasure. Every case remains in its original denominator. Strict
canonical mismatch remains separate from factual or semantic error.

## Actual states and complete evidence

The [registered protocol](glyph_tail_native_reader_protocol.md) fixes both
presentations, all 32 questions and their erased controls, both adapter
conditions, and separate complete-text controls. The
[registration](evidence/glyph_tail_native_reader_registration_20261111.json)
has SHA-256 `c6ae3bd9c56f3b933cb64ea0fcf391d392284b054c0830572096718bdef0e657`.
The exact adapted observer geometry, compiled numeric task frame, positional
convention and deterministic numerics reproduce each actual actor prefill
before native decoding. Original-adapter-zero calls are separate controls.
One body is reused serially; shared weights are not duplicated.

The source vocabulary projection, tokenizer and previous-symbol embedding
lookup are temporary helper variables outside the image actor. Source units
are not ILM glyphs. Both use a 128-unit cap in their respective representations,
so lengths and latencies are not matched. No source lexical module is attached
to the ILM, and no diagnostic output or development state enters training.
This helper cannot replace the requested independent image-output model.

The [authenticated result](evidence/glyph_tail_native_reader_learning_20261111.json)
retains all 448 unchanged decoded texts, stops/caps, output hashes, metric
denominators and paired intervals. Original source IDs, selected logits and
all contextual states remain hash-bound in local output archives. CPU
verification checks source IDs, end decisions, exact tokenizer decoding,
saved state coverage and the original glyph-control score bindings. The first
analysis stops on an older receipt-schema mismatch before saving scores; a
corrected lookup authenticates the same original hash. No generated case,
reference or scoring rule is changed, and generation is not repeated.

## Writing/feedback and layout contrasts

For adapted proportional input, native-versus-glyph exact QA is higher by
12.5 percentage points, with pointwise 95% passage-group bootstrap interval
**0–25 points**. The paired difference is 6.25 points, with interval **0–18.75**.
Seven cases are exact only through native decoding, three only through glyph
output, and three through both. Six of the seven native-only cases have an
uncertain whole glyph reading; the seventh is a readable instrument list
missing the canonical alias. The three glyph-only exact matches correspond
to native wording/punctuation differences: an added athlete noun, terminal
punctuation and an added party-membership noun. They are not adjudicated
native semantic errors. No post-hoc semantic accuracy is substituted.

The previously lost area answer is exact through both proportional native
conditions while its proportional glyph stream remains unreadable. This
demonstrates that this answer is available through the native continuation
path from the matched adapted image state. It does not identify whether the
first glyph, later writing, stopping or generated-image feedback fails.

Presentation strongly affects the native reader too. Proportional-minus-fixed
exact QA is **18.75 points [6.25, 31.25]** with adapted weights and
**34.375 points [21.875, 50]** with source adapters zeroed. Thus image-layout
brittleness is not confined to the glyph output head. The adapted proportional
image/text contrast is **minus 3.125 points [minus 15.625, 9.375]** when expressed
as image minus text; this interval includes zero. It is not proof of equivalent
capability. The source proportional score also exceeds its text score on this
small strict-match cohort, without demonstrating that images generally
outperform text.

Adapter effects are mixed. Adaptation changes proportional native exact QA
from 13/32 to 10/32, a difference of **minus 9.375 points [minus 21.875, 3.125]**,
while fixed-cell exact QA rises from 2/32 to 4/32 and text from 10/32 to 11/32.
Adapted fixed-cell native generation has thirteen caps versus one with source
adapters zeroed; adapted proportional generation has two versus one. These
controls do not justify globally removing the learned adapters, because no
corresponding unadapted glyph actor was trained or evaluated here.

All intervals resample the same sixteen whole passage groups with 10,000
draws and seed 202611154. They are pointwise development intervals, without
untouched confirmation or independent training-seed replication. Source
pretraining exposure remains unknown. The earlier glyph readings use one
model reviewer with prior cohort familiarity, without independent human
adjudication. Zero erased observations do not prove zero population success.

## Cost and next decision

The single model attempt takes **2,838.92 seconds**, including **158.82 seconds
setup** and **180.17 seconds pressure waiting**, with **18.947 GiB** peak
allocated GPU memory and at least **50.69 GiB** host memory available. The
temporary head is stored in BF16 and projected in ascending 4,096-row FP32
chunks with the registered tie rule. The run shares the workstation with
other workloads and bounded CPU publication assembly. It is not a controlled
matched-quality speed comparison. The worker has exited and GPU 0 is released.

The result supports retained native image-answer access and an additional
glyph-writing/feedback limitation, alongside substantial layout sensitivity.
It does not prove that a new writer, more training or a different sampler will
repair the model. Before another fit, inspect all current first glyphs and
target-form coverage, using authenticated original pixels and CPU-only
reconstruction controls. Keep the full complete-answer criteria unchanged.
The [next bounded decision](glyph_tail_first_writing_decision.md) specifies
that cheap localization. The usable independent visual-language goal remains
active and unachieved. These new helper results are outside the completed
29/51/51-page manuscript revision until a separate evidence update.
