# Exact sampled training writing does not improve complete-answer accuracy

The frozen balanced writer fails the registered current-cohort language
pilot. It gives 2/32 strict exact QA answers with fixed cells and 6/32 with
proportional prompts. Its sampled training glyphs were all pixel-exact, but
that physical qualification does not establish useful autonomous language.
The model and full ILM goal remain unqualified.

All 216 unchanged streams are complete. The checkpoint is fixed before the
trial; there is no new fit, checkpoint selection, sampling seed or best-of
selection. The trained readout and stopping predictor differ together from
the parent. Reader weights, contextual adapters, image encoding/decoding and
normalization remain frozen. All 216 actual initial reader states match the
parent bit-for-bit. This is a bundle comparison, not an isolated writer effect.

| Normal QA | Parent exact / readable / pairs | Original flow exact / readable / pairs | Balanced exact / readable / pairs |
|---|---:|---:|---:|
| Fixed cells | 2 / 3 / 1 | 2 / 4 / 1 | 2 / 3 / 1 |
| Proportional | 6 / 11 / 0 | 7 / 14 / 1 | 6 / 16 / 0 |

Each layout includes 32 questions from sixteen passage pairs. Both-correct
requires distinct actual output images and exact complete answers to both
questions. Every complete exact match requires natural stopping and the
unchanged whitespace-normalized reference rule. A canonical mismatch is not
by itself semantic falsity; no favorable semantic score replaces this rule.
Greater proportional readability is a limited observed change, without a
complete-answer accuracy gain or a correct paired question group.

All two fixed and six proportional parent exact answers are retained. Against
the original flow, 5/7 proportional exact answers remain: `qa_09_a_normal`
and `qa_15_b_normal` are lost, while the parent's `qa_10_a_normal` is recovered.
The union requirement retains only 6/8 proportional cases. The prospective
pilot requires at least ten exact proportional answers, two both-correct
pairs, sixteen readable answers and complete union/fixed retention. It fails.

Fixed normal QA has six stops and 26 caps. Proportional normal QA has eighteen
stops and fourteen caps. All twelve normal book prefixes have zero credited
readable, coherent complete continuations; four stop and eight cap. All
twelve erased book prefixes stop after unreadable output. All four QA-erasure
arms have zero complete exact answers. One proportional question-erased
output is readable, with zero exact matches; the other three arms have zero
readable outputs. Stopping or emitting a recognizable initial fragment alone
does not qualify a complete response.

The proportional exact rate is 18.75%, with pointwise 95% passage-group
bootstrap interval 6.25–31.25%. Compared with the original flow, the difference
is minus 3.125 percentage points, interval minus 12.5 to 6.25. The paired
difference is minus 6.25 points, interval minus 18.75 to zero. Exact outcomes
are identical to the parent on this sample, giving a degenerate zero
bootstrap difference; this does not establish population equivalence.
Intervals use 10,000 draws, seed 202611154 and sixteen groups. There is no
untouched confirmation, independent seed or independent human adjudication.
Twelve book prefixes from two works receive no broad generalization interval.

All 54 opaque boards and twelve original book prefixes are inspected before
the new case mapping and reference scoring. The 216 frozen readings contain
twenty clear and 196 uncertain whole transcriptions. One model reviewer has
prior cohort familiarity. Ambiguous readings retain uncertainty, including
legible fragments followed by damaged output. Nearest-neighbor enlargements
repeat original pixels without smoothing or repair. They resolve a few
transcriptions before freezing, including a visible 筒 substitution in an
instrument parenthetical. Scores are recounted in a separate invocation and
independently aggregated from per-answer records. All initial states are
compared independently with authenticated original tensors.

Fixed/proportional generation takes 1,534.527/1,140.482 seconds, including
82.055/77.201 seconds loading. Both use 17.786 GiB peak allocated GPU memory,
with zero recorded pressure waiting. Actual generation sums to
1,426.712/1,041.959 seconds. These costs exclude the inherited 7.522B visual
and contextual body training, the earlier 18,000-update balanced fit, offline
preparation and CPU/review work. They do not establish a foundation-training
or matched-quality efficiency advantage. Both workers exit and GPU0 returns
to its idle allocation; foreign GPU1 services remain untouched. No new
128-glyph book cold replay is run, and the earlier different-cohort replay is
not substituted.

The comparison shows that faithful writing on selected training states is
insufficient for this recipe's autonomous language transfer. It does not
identify a unique cause of the development gap. Preserve the sampled training
pass, development first-image failures, all earlier exact cases and complete
raw failures. Do not extend either failed rollout arm or start another
writer-only fit from these scores. The next model decision must localize
conditional transfer, reader access, generated-history drift and stopping
from these saved trajectories and the existing native control before choosing
a different bounded training mechanism.

The [prospective decision](balanced_writer_matched_decision.md),
[complete language record](evidence/balanced_writer_matched_learning_20261111.json),
[all-output review](evidence/balanced_writer_matched_20261111/review.json) and
[public export hashes](evidence/balanced_writer_matched_review_export_20261111.json)
bind all 216 outputs, 54 boards and twelve book prefixes. The
[physical diagnosis](balanced_writer_matched_diagnosis_result.md) remains
separate. The [bounded manuscript plan](manuscript_realized_writer_revision_plan.md)
will combine this result with the completed rollout pair in Figure 22.
Its source and data loader are implemented and the physical arrays recounted;
the native figure export and matching PDFs are pending. The verified
33/58/59-page MotorDiagnostics manuscript remains the current PDF baseline.
