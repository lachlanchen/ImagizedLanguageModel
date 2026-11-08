# Final-image continuation has a limited advantage over additional assisted training

The complete additional-training control gives **5/32 seen and 10/32 held-out
OCR-exact copies**, versus **15/32 and 13/32** in the final-image pilot. The
comparison meets its predeclared exploratory rule of three additional held-out
observations with no seen regression. **Held-out Chinese remains 3/16 in both
arms.** All three net held-out gains are English. Neither arm qualifies as a
reliable writer or a language model.

The [protocol](autonomous_writing_control_protocol.md) was written after the
pilot result was known and before the control was run. This is a retrospective
pilot with a newly specified control, not a prospectively randomized comparison.
Both continue the same starting weights for 32 updates, matching example
exposure, fresh optimizer settings and native input-encoding precision. The
old objective still uses assisted flow matching and gated reader loss; the
pilot uses full-trajectory final-image supervision and different loss weights.
The comparison does not isolate trajectory differentiation or match compute.

| Complete independent OCR observation | Old-objective control | Final-image pilot |
| --- | ---: | ---: |
| Seen, all | 5/32 | 15/32 |
| Held out, all | 10/32 | 13/32 |
| Seen Chinese | 0/16 | 3/16 |
| Held-out Chinese | 3/16 | 3/16 |
| Seen English | 5/16 | 12/16 |
| Held-out English | 7/16 | 10/16 |

Ten seen cases are OCR exact only in the pilot; none is exact only in the
control. Held out, four are exact only in the pilot and one only in the
control. The latter five cases are all English. For example, held-out 001
restores the missing `n` in `open` and 019 removes the repeated `r` in
`bright`. The two outputs in 007 look similarly readable despite different
OCR decisions, so every score change is not necessarily a content correction.
The clean-reference evaluator scores remain 27/32 and 28/32. OCR scores are
observations, not manual reading accuracy.

All 64 original-order targets, control pages and pilot pages are available:
[seen 00–15](evidence/autonomous_writing_control_result/all_cases_00_15.png),
[seen 16–31](evidence/autonomous_writing_control_result/all_cases_16_31.png),
[held-out 00–15](evidence/autonomous_writing_control_result/all_cases_32_47.png),
[held-out 16–31](evidence/autonomous_writing_control_result/all_cases_48_63.png).
The [complete export](evidence/autonomous_writing_control_result/result.json)
retains every transcript, reference/input binding and actual neural tensor
hash. Each control PNG equals its quantized PT pixels. There is no rendering,
OCR repair or quality-selected reroll. This authenticates saved artifacts,
not an independent neural replay of all outputs.

Visible failures reject the unchanged 30/32 writing floor. In control, seen
000 adds unwanted marks, 011 duplicates its line, and 012/015/031 copy part
of the instruction. Held-out 002 repeats a character, 026 duplicates its
line, and 030 copies the instruction. The pilot's omissions, duplicated
line and instruction copying remain visible alongside its improvements.
Some apparently legible Chinese is underrecognized, but that does not erase
the visible errors or establish answer correctness.

## Execution and provenance

The first Chinese and English starting outputs replay bit-exactly before
optimization. All 32 IDs and seed keys reproduce the pilot journal and its
registered CPU RNG streams. Both visit 22 distinct examples: 13 Chinese
draws cover nine distinct Chinese inputs; 19 English draws cover thirteen
distinct English inputs. Control recognition is active on 15 updates.
Student gradients are finite and nonzero, all 206 adapter tensors change,
and base, VAE and reader weights remain frozen. All 64 final native calls use
the original input-only records and seeds, with no inference-time teacher,
target or lexical output head.

| Measured control stage | Seconds | Peak allocated GPU memory (GiB) |
| --- | ---: | ---: |
| Complete training invocation, including two initial replay calls | 56.962 | 12.251 |
| Optimization loop within that invocation | 33.790 | Not separately reset |
| Two initial replay calls within that invocation | 16.192 | Not separately reset |
| Complete final 64-case native inference | 513.381 | 8.031 |

The previous final-image training invocation takes 1,135.452 seconds at
12.057 GiB. Instrumentation and initial replay work differ. Costs exclude
external pretraining, earlier optimization, separate CPU assessment and an
interrupted inference invocation. There is no quality-matched language
efficiency claim. Both deployed transformers have 3,899,903,488 parameters
plus an 84,046,115-parameter VAE; 24,358,912 adapter parameters are trained.
The 560,528,640-parameter reader is training only.

An initial control-inference invocation was stopped after 19 completed
outputs because an unrelated exporter commit changed the global Git identity
required by the legacy runner. Its files remain preserved. The replacement
uses identical weights, inputs, seeds and settings; all nineteen overlapping
PNGs and PT pixel tensors reproduce exactly. This was an execution-identity
recovery, not output selection. The exported interruption receipt records it.
The completed invocation's source commit is fixed throughout generation.

## Consequence for the ILM goal

The result supports a limited next learning experiment with the final-image
objective bundle. It does not establish a unique causal repair, an efficient
training algorithm, generalization across seeds or improved held-out Chinese.
Natural QA has not been optimized or assessed in either continuation. More
accurate copying, prompt-dependent answers and wider writing forms remain
necessary to establish the requested independent glyph-based language model.
Physics analogies are not used as evidence of those capabilities.

Reproduction uses `scripts/train_autonomous_writing_control.py`, the unchanged
`scripts/infer_paired_image_cohort.py`, `scripts/score_paired_writing.py` and
`scripts/export_autonomous_writing_control.py`. Existing pilot sources remain
unchanged. A separately specified balanced-exposure continuation can test
whether the current writer learns further; it is not yet a result.
