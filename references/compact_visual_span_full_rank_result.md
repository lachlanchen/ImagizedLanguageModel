# Wider visual writer: complete language result

Widening the writer removed the measured output-rank restriction but did not
produce a credited readable complete answer. All five checkpoints scored
0/32 complete exact QA answers and 0/16 both-correct question pairs on the
registered development cohort. No readable complete book continuation was
credited. The final comparison with compact V1 is zero on both QA metrics.
The usable visual-language-model goal remains unachieved.

## Controlled comparison

The [registered protocol](compact_visual_span_full_rank_protocol.md) changes
the conditional writer width from 384 to 1,024, retaining the 768-coordinate
codec, encoder, contextual foundation, supervision, corpus, optimizer recipe
and evaluation. Both fits expose all 13,785 training sequences and make 8,172
updates. The wider model has 48,547,733 trainable image parameters and an
additional 8,388,608 training-only projection parameters. It retains the
frozen 1.409B externally pretrained contextual body and fixed project codec.
This tests visual adaptation of inherited language capability.

The trained output projection, LayerNorm-centered output map and three tested
local velocity Jacobians have rank 768. These finite capacity checks do not
establish learnability, global invertibility or output quality. The wider fit
took 1,945.56 seconds, including 150.14 seconds of resource waiting, and peaked
at 4.889 GiB allocated GPU memory. See the
[complete fit receipt](evidence/compact_visual_span_full_rank_completed_20261111.json).

## Fixed outputs and review

The unchanged suite contains 32 questions from sixteen held-out passage
groups and twelve prefixes from two held-out books. Normal prompts run at
five fixed steps. Both endpoints also run the erased-input controls, giving
372 answer images. Generation uses seed 202611153, eight flow steps, a
128-glyph budget, learned boundaries, lengths and stopping, and actual emitted
pixels as subsequent history. Inherited pretraining exposure is unknown.

| Update | Exact QA / 32 | Both-correct pairs / 16 | Readable complete books / 12 | Normal stops / 44 | Empty normal outputs / 44 | Capped normal outputs / 44 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 | 40 | 9 | 4 |
| 1,956 | 0 | 0 | 0 | 14 | 7 | 30 |
| 4,049 | 0 | 0 | 0 | 19 | 3 | 25 |
| 6,152 | 0 | 0 | 0 | 23 | 1 | 21 |
| 8,172 | 0 | 0 | 0 | 13 | 4 | 31 |

One Codex reviewer inspected all images under opaque identifiers on 31
boards, with exact nearest-neighbour display zoom and verified original
hashes. The case/checkpoint mapping and scoring references were not opened
during transcription. The reviewer had prior dataset familiarity; there was
no independent human adjudication. Twenty-four images were blank. The other
348 contained uncertain whole-response transcriptions. Individual characters,
digits and punctuation were sometimes recognizable, but damaged or ambiguous
glyph identities prevented reliable complete readings. Several repeated
strings were visible. Uncertainty concerns glyph identity; an otherwise
legible nonsensical string would still require transcription.

Readings were frozen before reference scoring. All endpoint QA controls also
score zero. Final-minus-initial, normal-minus-erased and final-minus-V1 exact
QA counts are zero. The registered sixteen-group bootstrap yields degenerate
[0, 0] sample intervals; it does not establish zero population success.
Book coherence is limited by uncertain transcription, rather than a separate
assessment of meaning behind unreadable marks. Only two held-out works are
represented, so no book confidence interval is reported.

The analyzer re-ran the scorer and authenticated the five checkpoints,
registered inputs and original output pixels. The
[result receipt](evidence/compact_visual_span_full_rank_learning_20261111.json)
binds the complete analysis and the
[frozen readings](evidence/compact_visual_span_full_rank_readings_20261111.json).
The same first QA pair and first book prefix, selected before training, have
all fifteen unrepaired output PNGs in the
[evidence directory](evidence/compact_visual_span_full_rank_learning_20261111/),
with [editable measurements](evidence/compact_visual_span_full_rank_learning_20261111/language_metrics.csv).

The unchanged native-text Qwen3 1.7B reference gives 8/32 exact QA answers and
2/16 both-correct pairs. It ran on CPU and supplies a task-quality reference;
matched-quality, matched-GPU efficiency remains unmeasured. The wider writer's
final 120-case generation takes 682.51 seconds plus 15.00 seconds loading,
with 3.347 GiB peak allocated GPU memory. Response lengths and stop decisions
differ, so elapsed times alone do not establish an efficiency advantage.

## Standalone boundary and decision

The trained cold child reproduces the fixed 128-glyph book stream, stop/cap
status and all 128 emitted-pixel feedback hashes. No forbidden training or
reference opens or IP connections were observed. Its wrapper terminated
before saving verification, for an unestablished reason. All retained child
thread traces end with exit zero; CPU-only verification recovered the receipt
without rerunning generation. This checks the standalone boundary for that
invocation, while its answer remains a language failure. See the
[cold receipt](evidence/compact_visual_span_full_rank_cold_20261111.json).

The rank restriction was real, but removing it was not sufficient under this
recipe and exposure. Preserve both fits and their raw failures. Before another
training change, measure visual access with known boundaries and counts,
conditional writing under reference-image history, the eight-step sampler
against a finer integration, and the effect of emitted-image history. Include
direct codec reconstruction and shuffled-context controls. Reference-supplied
diagnostics must remain separate from autonomous language scores.

The next step is [failure localization](compact_visual_span_failure_localization_plan.md).
No longer fit, new architecture, larger foundation or new data collection is
selected by these results. The measurements must identify the next bounded
intervention; a plausible mechanism alone is insufficient.
