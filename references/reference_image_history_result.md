# Correct image history improves later writing but does not qualify the ILM

The frozen comparison is complete. All 64 normal QA traces, 1,490 reference
positions and 64 terminal boundaries are generated and independently recounted.
All sixteen boards were actually viewed, and all 8,561 displayed original tiles
match their bound arrays. No optimizer update, new autonomous answer or native
helper rollout was made. The registered language scores remain unchanged.

The [prospective decision](reference_image_history_decision.md),
[registration](evidence/reference_image_history_registration_20261112.json),
[complete measurements](evidence/reference_image_history_learning_20261112.json)
and [completed review](evidence/reference_image_history_review_20261112.json)
retain every case. The initial analysis receipt predates visual review; the
separate review receipt records its completion and binds the original receipt.

## Conditional writing

The model predicts at each reference boundary while receiving only the correct
preceding glyph images. Stopping is recorded but bypassed, and the reference
length is supplied. This is a diagnosis with answer information in its history,
not autonomous language generation or evidence of independent comprehension.

| Layout | First mean F1 / 32 | Later median F1 / 713 | Later positions at F1 0.95 | Later exact bitmaps |
| --- | ---: | ---: | ---: | ---: |
| Fixed cells | 0.479076 | 0.573840 | 139/713 | 59/713 |
| Proportional | 0.658382 | 0.985185 | 475/713 | 251/713 |

Both registered later-writing gates fail: neither layout reaches both median
F1 0.99 and 90% of positions at F1 0.95. First images are 0/32 and 4/32
pixel-exact. Equal-case mean F1 across complete references is 0.641395 fixed
and 0.914527 proportional. These are canonical bitmap measurements, not language
or character accuracies. Spaces and punctuation remain in their original pools.

At corresponding positions present in both histories, proportional mean F1 is
0.912711 with correct prior images and 0.445851 with saved autonomous images
(598 positions). Fixed-cell means are 0.614716 and 0.393464 (738 positions).
The 147 missing proportional and seven missing fixed autonomous target positions
remain missing; extra autonomous positions are also retained. Alternate wording
and lengths mean these comparisons are not semantic alignments or estimates
of a language-accuracy improvement.

Visual review shows substantially clearer later proportional writing under
correct history, including names, a route explanation and classical quotations.
Initial forms and some interior forms remain damaged or differ from the
reference. Fixed-cell conditional writing often remains malformed. Existing
readable alternative answers remain in their original images and readings.
No transcription is repaired and no new semantic score is substituted.

## Stopping and state identity

Relative to the supplied reference boundaries, fixed-cell prediction crosses
0.5 at 27/745 nonterminal positions in eleven cases and recognizes 24/32
terminals. Proportional prediction crosses at 14/745 positions in nine cases
and recognizes 30/32 terminals. The fixed stopping diagnostic fails; proportional
passes its registered position-rate/terminal-rate rule. This does not establish
natural autonomous stopping: both the history and terminal location were supplied,
and almost a third of proportional cases contain an early reference-relative stop.

All 64 initial states are bit-exact with saved autonomous execution. Every saved
autonomous emitted pixel and stop probability is reproduced from its saved state
by the same frozen bridge. All parameter versions remain unchanged.

The additional common-prefix analysis is post hoc and descriptive. All 64 traces
have identical states while their preceding images are identical. In the 63
cases with a differing image, the first changed state follows immediately after
that image. This verifies dependence on image history; it does not label that
change harmful. In particular, all four cases with a first-image F1 of at least
0.95 but nonidentical pixels have readable original autonomous answers:
`fixed_cells__qa_05_b_normal`, `proportional__qa_07_a_normal`,
`proportional__qa_07_b_normal` and `proportional__qa_16_b_normal`.
The exact first F1 values and all per-case means are in the review receipt.
A universal explanation based on small pixel errors causing collapse is therefore
unsupported by this observation.

## Decision

A feedback-only explanation is insufficient: conditional writing already fails
before generated history and remains imperfect with correct history. Improving
first and later conditional transfer is necessary for this recipe; the supplied
prefix benefit alone does not identify a unique repair. State divergence and
canonical mismatch cannot substitute for complete-answer evidence.

The earlier [feedback ceiling](glyph_feedback_ceiling_result.md) also remains
relevant: on its different cohort, a source policy gives 12/16 literal native-unit
matches but only 3/16 with exact-character or canonical-image feedback. Taking
one character of a native unit does not preserve its original sequence dynamics.
Do not repeat that study, count its helper answers as ILM output, or train from
its excluded diagnostic records. The earlier mixture-density, function-transfer,
compact-span, tail-adaptation and flow failures remain controls.

The [next direction](generation_only_adaptation_direction.md) addresses the
visual continuation dynamics while preserving source prompt prefilling. It
requires a real memory/causality/gradient preflight before a registered full fit.
It is not implemented or qualified by this result. Another extension of the
failed writer-only fits is not selected.

## Cost and limits

Execution takes 460.834 seconds, including 70.067 seconds of pressure waiting,
with 17.786 GiB peak allocated VRAM and at least 55.714 GiB available host RAM.
Loading is recorded as 137.167 seconds; loading and pressure intervals may overlap.
There are 1,554 new contextual calls. The same worker completes and exits with
code zero; GPU0 returns to 16 MiB, and GPU1's existing service is untouched.
The total inherited visual/contextual body remains 7,522,463,984 parameters.
This is neither a compact-model nor a matched-quality efficiency result.

All cases are reused development data. One model reviewer has prior cohort
familiarity; independent human adjudication and untouched confirmation are
absent. The usable, independently running visual-language goal remains active
and unachieved, including modern language, book continuation and later historical
and unencoded writing.
