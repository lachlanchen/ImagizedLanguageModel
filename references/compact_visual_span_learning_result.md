# Compact visual span model: complete-sequence result

The compact V1 model did not produce a credited readable complete response on
the registered development cohort. Its five checkpoints each scored 0/32
complete exact QA answers, 0/16 both-correct question pairs and 0/12 readable
complete book continuations. The fit and inference ran successfully; those
execution results do not establish a working visual language model.

## Training and fixed evaluation

The fit completed 8,172 optimizer updates on 13,785 complete Chinese sequences,
covering 2,576,239 target glyph positions and 2,380,620 causal state targets.
It used the frozen 1.409B contextual foundation, fixed V34 codec and 12.87M
trainable image parameters, with an additional 8.39M training-only teacher
projection. Training used reference-image history and offline native alignment.
Generation used learned boundaries, lengths and stopping, eight flow steps,
seed 202611153 and the model's own emitted pixels as subsequent input. The
output budget was 128 glyphs. These components transfer inherited language
knowledge to a visual interface; this experiment does not test language
acquisition from images alone.

All 44 normal prompts ran at every fixed checkpoint. Both endpoints also ran
the 76 erased-input controls, giving 372 generated pages. The protocol retained
all failures, blank responses and capped streams. Source partitions hold out
whole works and CMRC passage groups; inherited pretraining exposure is unknown.
See the [registered evaluation](compact_visual_span_evaluation_protocol.md) and
[analysis procedure](compact_visual_span_learning_analysis.md).

| Update | Exact QA / 32 | Both-correct pairs / 16 | Readable complete books / 12 | Normal stops / 44 | Empty normal outputs / 44 | Capped normal outputs / 44 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 | 44 | 9 | 0 |
| 1,956 | 0 | 0 | 0 | 44 | 9 | 0 |
| 4,049 | 0 | 0 | 0 | 38 | 8 | 6 |
| 6,152 | 0 | 0 | 0 | 23 | 2 | 21 |
| 8,172 | 0 | 0 | 0 | 17 | 2 | 27 |

The endpoint question-erased and passage-erased QA controls also scored zero.
The final-minus-initial and normal-minus-erased complete-answer differences
are therefore zero. The registered passage-group bootstrap gives degenerate
[0, 0] intervals for these constant observations. This describes the sample,
not a population success probability of exactly zero. Only two book works
were evaluated, so no book-level confidence interval is reported.

## Visual review and its limits

One Codex reviewer inspected all 372 answer images under opaque identifiers
before opening their case/checkpoint mapping or scoring references. Displays
used exact nearest-neighbour scaling with verified round trips; original image
hashes were rechecked. The reviewer had prior familiarity with the dataset.
There was no independent human adjudication.

Thirty pages were blank. The other 342 had uncertain whole-output
transcriptions: individual Chinese characters, punctuation and digits were
sometimes recognizable, but fragmented or ambiguous marks prevented reliable
complete readings. Several pages repeated similar strings. The reviewer
recorded uncertainty rather than replacing ambiguous glyphs with plausible
Unicode text. The complete-response metric credits none of these outputs as
readable. Book coherence was consequently not established; the result does
not independently assess meaning hidden behind an unreadable transcription.

All readings were frozen before reference scoring. The analyzer recomputed
the scores, authenticated input/checkpoint bindings and original pixel streams,
and exported the fixed examples. The first QA pair and first book prefix were
selected before fitting. Their fifteen raw answer PNGs are retained in the
[evidence directory](evidence/compact_visual_span_learning_20261111/), with
[editable chart data](evidence/compact_visual_span_learning_20261111/language_metrics.csv).
The [result receipt](evidence/compact_visual_span_learning_20261111.json) binds
the checkpoints, scores, readings and image hashes. The
[review record](evidence/compact_visual_span_readings_20261111.json) preserves
the uncertain and blank readings.

The unchanged Qwen3 1.7B native-text reference scored 8/32 exact QA answers and
both answers in 2/16 pairs on the same normal prompts. Its book coherence is
unrated. It ran on CPU, so this is a task-quality reference rather than a
matched-hardware efficiency result. See the
[reference uncertainty analysis](evidence/compact_visual_span_qwen_uncertainty_20261111.json).

## Standalone boundary and resource use

The complete fit took 1,759.83 seconds, including 110.11 seconds waiting for
host memory pressure, and peaked at 3.930 GiB allocated GPU memory. Offline
teacher extraction was a separate completed stage, taking 1,783.41 seconds
including 770.72 seconds of resource waiting and peaking at 13.396 GiB allocated
GPU memory. These are adaptation and preparation costs, not from-scratch
training costs for the inherited contextual foundation.

Each inference stage peaked at 3.213 GiB allocated GPU memory. The final
120-case generation took 804.84 seconds, plus 19.67 seconds loading its
checkpoint. Raw generation times vary with emitted length and stopping. No
matched-quality efficiency advantage is established by these measurements.

A separate traced process loaded the final checkpoint and reproduced the fixed
128-glyph book answer, its PNG and pixel stream, stop/cap status and all 128
emitted-pixel feedback events. Required weight and prompt opens were verified;
no forbidden training/reference paths or IP connection attempts were observed.
The receipt includes all trace hashes. This establishes the standalone boundary
for that invocation, while its generated answer remains a language failure.

## Decision

The [flow-head audit](compact_visual_flow_rank_result.md) identifies an
architectural restriction in V1: its linear velocity output has rank 384 in a
768-dimensional codec space. In the continuous model, the complementary 384
Gaussian directions cannot be contracted by that velocity field. The audit
does not establish the sole cause of these failed responses or quantify the
resulting pixel error.

Preserve V1, its loader and all outputs. The next model comparison tests a
full-dimensional velocity path with the same encoder, contextual foundation,
codec, corpus, supervision, optimizer settings and prompt cohort. Register its
implementation and resource budget before fitting. Evaluate the actual answer
images and question dependence again. If writing remains unreadable, localize
visual-input alignment, conditional writing and generated-history drift before
choosing another change. Additional epochs of the same narrow writer are not
the next experiment. The usable-language goal remains unachieved.
