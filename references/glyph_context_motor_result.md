# Actual-context motor: complete language comparison

The direct conditional motor fails its registered development pilot. All 216
generated streams,54 opaque boards and 12 book prefixes were reviewed before
reference scoring. The frozen readings contain 20 clear whole outputs and 196
uncertain whole readings. This is one model review with prior cohort
familiarity and no independent human adjudication.

| Normal-input measure | Fixed parent | Fixed motor | Proportional parent | Proportional motor |
|---|---:|---:|---:|---:|
| Complete exact QA /32 | 2 | 2 | 6 | 7 |
| Both answers correct /16 passage pairs | 1 | 1 | 0 | 1 |
| Readable whole QA /32 | 3 | 4 | 11 | 14 |
| Naturally stopped QA /32 | 8 | 4 | 19 | 18 |
| Credited complete book continuation /12 | 0 | 0 | Not run | Not run |

The fixed candidate retains both original exact answers. The proportional
candidate retains five of six, loses `qa_10_a_normal`, and gains the area and
sports-list answers. Its pilot requires at least 8/32 exact answers, 2/16
correct pairs, 11/32 readable outputs and retention of all six proportional
and both fixed exact answers. It fails the exact, paired and proportional
retention criteria. There is no untouched confirmation or usable-ILM result.

The proportional exact difference is 3.125 percentage points, with a
pointwise 95% passage-group bootstrap interval of minus 6.25 to 12.5 points.
The paired difference is 6.25 points [0,18.75]. These use 10,000 resamples of
sixteen paired passage groups. They describe a small reused development
cohort, have no simultaneous-coverage guarantee, and do not establish a
reliable general improvement. Identical fixed exact counts give a degenerate
difference interval without proving equal population performance. Twelve book
prefixes from only two works receive no generalization interval.

All QA-erasure arms have zero exact answers. Each proportional erasure arm has
one readable short output; readability alone does not establish an answer.
No normal book output is credited as readable, coherent and complete. Strict
canonical mismatch does not itself establish semantic or factual falsity.
Uncertain partial readings are not promoted after references are opened.

The 24,477,696-parameter head maps actual 4096-wide reader states to the 1024-wide
continuous motor, bypassing the old semantic point. It preserves the reader,
adapters, image encoder, neural decoder and stop predictor. Training uses 3,176
first and 20,086 later authenticated training states, with 768,000 exposures of
each pool over 12,000 updates. The current development answers and helper
diagnosis are excluded from fitting. Final EMA, per-case seeds, one sample per
glyph, 32 integration intervals and the 128-glyph cap remain fixed.

Independent CPU comparison finds all 216 initial contextual states bit-for-bit
equal to their matched parent states from the same physical images. Output
changes therefore begin after an unchanged initial reader state. This does
not separate later motor prediction, generated-history drift and stopping:
the candidate rereads its own changed pixels. Nor does it repair measured
input-layout brittleness or establish language acquisition from images alone.

The head fits in 109.485 seconds at 0.989 GiB peak allocated GPU memory. Full
generation takes 3,442.507 seconds for 120 fixed streams and 2,720.582 seconds
for 96 proportional streams, at 17.777 GiB peak allocation. Their endpoint
counts are 12 stops/108 caps and 25 stops/71 caps. Full deployments retain 7.522B
inherited visual/contextual parameters. Small head-fitting cost is not the
cost of training a foundation model. Setup failures, offline preparation and
inherited pretraining remain separate costs; no task-efficiency advantage is
shown.

The fixed `fixed_cells__book_01_normal` replay initially reproduces all nested
values and PNG bytes but fails its access gate: shared training admission
opens old ownership ledgers. That 125.884-second attempt is retained.
Standalone admission is repaired to check live processes, the GPU mutex,
memory and pressure without training-ledger access. The same input, weights,
seed, cap and sampler then reproduce every pixel, contextual state, stop
probability, feedback hash, sampled noise and motor value. Its trace shows no
forbidden training/reference artifact opens or IP connections. It takes 100.755
seconds at 17.748 GiB peak allocation. The source archive's unused vocabulary
payload remains disclosed: file traces alone cannot establish tensor-level
exclusion, which also depends on the bound loader and runtime provenance.
The reproduced book output remains capped and unreadable. This is a passed
execution check, not a language success.

The [complete evidence](evidence/glyph_context_motor_learning_20261111.json)
records the recount, intervals, retention, initial-state comparison, source
hashes, resources and both replay attempts. See the
[registered protocol](glyph_context_motor_protocol.md),
[fit result](glyph_context_motor_fit_result.md) and
[cold recovery](evidence/glyph_context_motor_cold_recovery_20261111.json).
