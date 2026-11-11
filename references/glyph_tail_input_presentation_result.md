# Full proportional prompts improve individual answers but lose a pair

With the same frozen actor and complete passage/question content, proportional
input presentation gives **6/32 complete exact QA answers**, versus **2/32**
for fixed glyph cells. Whole-response readability increases from **3/32 to
11/32**, and normal QA caps decrease from **24/32 to 13/32**. However,
both-correct passage pairs fall from **1/16 to 0/16**. The previously exact
area answer becomes unreadable. The registered retention rule fails, so this
presentation is not adopted as a uniformly better interface.

| Normal QA measurement | Fixed cells | Proportional raster |
| --- | ---: | ---: |
| Complete exact stopped answers | 2/32 | 6/32 |
| Both distinct questions exact | 1/16 | 0/16 |
| Complete readable outputs | 3/32 | 11/32 |
| Predicted stops | 8/32 | 19/32 |
| Capped streams | 24/32 | 13/32 |
| Empty outputs | 0/32 | 0/32 |

These are frozen-inference transfer results, with zero new optimizer updates.
They are not a new language-learning curve or a usable ILM qualification.
The retained body has 7.522B inherited visual/contextual parameters. The
font face, spacing and wrapping change together; the comparison does not
isolate one of those causes. Book continuation is not tested by this QA-only
comparison. Both compact failed fits and the earlier fixed-cell control remain
unchanged.

## Complete inputs and independent review identities

The [registered protocol](glyph_tail_input_presentation_protocol.md) uses all
32 current questions and their 64 erased-input controls. CPU preparation
authenticates every full source prompt, preserves every non-whitespace
codepoint and changes each original newline into one ordinary space. The
exact earlier proportional renderer and font/runtime provenance match the
adaptation-data identity. Capacity comes from the full measured prompt,
without answer-centered cropping, truncation or font shrinking. Median page
height changes from 704 to 512 pixels at 1,024-pixel width.

Color-isolated passage, question and structural masks must be disjoint and
reconstruct every original ink pixel. Erased controls whiten those original
pixels in place, retaining the normal page's wrap ranges and geometry. All
96 images invert exactly to their complete source strips and pass the
unchanged host-preparation pixel check. The
[CPU rendering receipt](evidence/glyph_tail_input_presentation_rendering_20261111.json)
and [registration](evidence/glyph_tail_input_presentation_registration_20261111.json)
were committed before neural execution.

The generic review helper would reuse the previous comparison's opaque IDs.
A separate CPU review preparation adds a presentation/registration namespace
before any new output inspection. One Codex model reviewer then inspects all
eight boards and enlarged ambiguous strings. The
[frozen readings](evidence/glyph_tail_input_presentation_learning_20261111/readings.json)
have SHA-256 `c0322b0f81a364738397b4dfe95238cddc3231a03ec7e0584159ce3c99d0820e`.
The mapping and references are opened only after freezing all 96 readings.
Eleven pages have complete confident readings; 85 are uncertain, with no
blank pages. Prior cohort and fixed-cell-output familiarity are disclosed.
There is no independent human adjudication.

The [complete authenticated result](evidence/glyph_tail_input_presentation_learning_20261111.json)
recomputes both comparisons' scores from frozen readings, verifies every
original signed/binary/PNG output and feedback event, and preserves all
denominators. The unchanged strict scorer requires a learned stop and
whitespace-normalized equality with the complete canonical reference.
Both-correct pairs also require distinct images. Question-erased and
passage-erased QA conditions each remain 0/32 exact. Their readable counts
are also zero. Ten question-erased and three passage-erased streams stop;
the remainder reach the cap.

## Uncertainty and answer retention

The proportional exact rate is 18.75%, with a pointwise 95% passage-group
bootstrap interval of **6.25–31.25%**. Its gain over the fixed-cell rate is
12.5 percentage points, with an interval of **0–25 points**. The paired
rate difference is minus 6.25 points, with an interval of **minus 18.75 to
zero**. Neither difference interval excludes zero. The small reused
development cohort supplies no untouched confirmation or independent training
seed replication. Constant zero observations give degenerate sample intervals,
without proving zero population success.

Two of the original three readable cases survive: the location answer
**湖南省长沙县中南部** remains exact, and the shorter instrument list remains
readable but noncanonical. The original exact **169平方公里** answer no longer
has a reliable whole transcription. Thus the increased aggregate exact count
does not retain the original correct pair. The conservative original-core
retention check was implemented during generation, before any new output
inspection; its timing is recorded rather than described as preregistered.
The primary pair/retention rule already fails without that extra check.

Strict mismatch is separate from factual or semantic error. Some readable
outputs omit a country name or parenthetical alias. The date-only answer
**1972年4月1日** differs from a full-sentence canonical annotation, and
**8.54万人** omits the annotation's population label and year. These are
not newly adjudicated semantic-accuracy estimates. In another case the
frozen reading gives district **248**, versus canonical **24B**, exposing a
digit/letter fidelity problem. No transcription is repaired after references
are opened, and the registered scores are not retroactively changed.
Future semantic assessments need a prospective rubric on new source groups.

The public [matched images](evidence/glyph_tail_input_presentation_learning_20261111/raw_examples/)
retain both original input presentations and both untouched answers for
thirteen cases: the first fixed pair, all three original readable cases and
all eleven new readable cases, with overlaps counted once. The new-readable
display selection is retrospective and explicitly labeled. All 96 streams
remain local and hash-bound; selected illustrations do not replace the full
scoring denominator.

## Execution cost and next decision

CPU preparation takes 4.16 seconds. The 96-case neural attempt takes
**1,179.29 seconds**, including 97.89 seconds loading and 1,060.67 seconds
generating, with **17.786 GiB** peak allocated GPU memory and no recorded
pressure wait. The earlier run includes additional book inputs, so its total
wall time is not a matched speed comparison. These costs exclude inherited
pretraining, original adaptation and corpus preparation.

The [fixed-question cold replay](evidence/glyph_tail_input_presentation_cold_20261111.json)
reproduces signed pixels, contextual states, stopping and all actual feedback
in a separate traced invocation, with no observed forbidden corpus/reference
or font access or IP connection. Its unchanged CLI displays one long row;
exact tensors and display pixels, rather than wrapped-PNG file bytes, are
compared. It takes 79.08 seconds at 17.693 GiB peak, including 70.29 seconds
loading and 7.84 seconds generating. This checks one independent invocation,
not language quality. Both model workers have exited and GPU 0 is released.

The result establishes input-presentation sensitivity and limited readable
transfer, without reliable paired answering or full retention. Before another
fit, register a helper-only native-reader comparison using the same actual
image prefills, with the source lexical decoder kept outside the ILM. It
must distinguish visual access, tail-adaptation effects and the glyph
writer/feedback path. The [next decision](glyph_tail_native_reader_decision.md)
specifies that boundary. The independent visual language goal stays active.
