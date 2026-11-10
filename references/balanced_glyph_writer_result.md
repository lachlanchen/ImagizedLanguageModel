# Better training fidelity does not improve new-context answers

The balanced writer refinement completes 18,000 additional updates from the
fixed contextual parent. It passes the unchanged physical qualification but
does not improve complete answers on a newly reserved context cohort. Keep
the parent as the language reference. The
[complete paired archive](evidence/balanced_glyph_writer_pair_v1/archive.json)
retains all 144 new autonomous image streams and all 52 review boards.

| Measure | Unchanged parent | Balanced writer |
|---|---:|---:|
| Training mean foreground F1 | 0.963524 | 0.999811 |
| First training glyph F1, 3,176 positions | 0.893582 | 0.999559 |
| Later reference-conditioned F1, 20,086 positions | 0.974584 | 0.999851 |
| Canonical exact first training glyphs | 995/3,176 | 3,111/3,176 |
| Terminal training decisions correct | 3,130/3,176 | 3,176/3,176 |
| Continuation training decisions correct | 23,030/23,262 | 23,262/23,262 |
| Held-wording copy F1 | 0.957366 | 0.961801 |
| Held copies reaching F1 0.95 | 208/256 | 214/256 |
| All physical gates passed | No | Yes |
| Clear complete matches, new questions | 19/32 | 18/32 |
| Both different questions correct within a new context | 6/16 | 5/16 |
| Canonical exact complete answers, new questions | 2/32 | 2/32 |
| New answers reaching the 64-glyph cap | 11/32 | 8/32 |

These thirty-two questions come from sixteen contexts excluded from training,
the earlier development manifest and the first reserved cohort. They were
selected by the registered hash ordering without answer-coverage or
teacher-success filtering. They are not the first trial's thirty-two cases.
All thirty-two input images were inspected before fitting. One question lacks
a fully supported canonical reference in the physical evaluator; it remains
in every answer denominator. Source pretraining exposure is unknown.

Both models share eighteen clear complete literal matches. A nineteenth is
clear in the parent and uncertain in the refined output because its first
glyph is damaged. No new question becomes a clear complete match under
refinement. This subjective one-case difference is not evidence of a
statistically established language regression. The unchanged canonical match
count and failed preset pilot also give no evidence of an answer-accuracy
gain. More natural stops can still end in damaged or incorrect answers.

The original sixteen development questions are now reused evidence: clear
matches rise from six to seven, but canonical exact matches fall from two
to one. Erasing those questions gives no clear correct answer in either
condition. All eight training controls become canonical image-exact after
refinement. These are distinct measurements, not a substituted success rubric.

Every one of the thirty-six output boards was actually viewed before either
new-reference score; nineteen uncertain stopped streams were additionally
viewed as unchanged nearest-neighbour enlargements. Readings are conservative
agent transcriptions, not independent human grades. No strokes, missing
forms or quotes were repaired from a question or reference. The preset
24/32-match and 10/16-context pilot fails in both conditions.

The experiment freezes the complete learned reader, its contextual deltas,
image encoder, decoder and normalization, while training only 17,848,833
writer/stopping parameters. It adds 864,000 first targets, 864,000 later targets
and 576,000 copy targets. Exposure and position weighting change together;
their effects are not causally separated. An independent tensor/optimizer
audit confirms 18,000 real writer updates, 13,500 stopping updates and
bit-identical frozen modules and inherited contextual adapters.

The actual fixed-reader cache reproduces every parent training pixel score
and stopping count. It takes 1,051.67 s, including 630.60 s pressure waiting,
at 2.168 GiB peak CUDA allocation. The subsequent fit takes 156.38 s, including
20.02 s pressure waiting, at 1.096 GiB. An initial strict resource admission
fails before model loading or any update and is preserved; the successful
retry uses the same recipe and limits. The unchanged 7.522B inherited body,
parent training, source-cache preparation and external pretraining remain part
of the method's cost.

A cold standalone replay on the new cohort's prospectively fixed first input
reproduces all actual pixels, states, stopping and feedback. The trace records
no prohibited bank/font/corpus access or IP connection attempt. It is one
runtime-independence check, not a correctness theorem.

This result resolves a concrete training-fit problem while leaving language
generalization and autonomous feedback unresolved. Further writer fitting on
the same fixed states is not justified as the sole remedy. Sequence alignment
and learning on actual generated histories require separate tests; no outcome
from those tests is claimed here.
