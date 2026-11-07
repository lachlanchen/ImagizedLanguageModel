# Physical supervision and erasure geometry

The independent CPU [audit](../scripts/audit_contextual_pixel_data.py)
authenticates 6,514 training input/reference files. All 3,257 reference PNGs
have an admissible exact occurrence in their input; there are 3,941 matching
source spans in total. Searching physical column bytes independently of the
training matcher preserves repeated complete-image occurrences. Training
article titles and paragraph hashes remain disjoint from development.
The [numeric record](evidence/contextual_pixel_data_audit.json) retains every
control case and file digest. No model is invoked or trained by this audit.

| Physical control, 64 cases each | Question erased | Passage erased |
| --- | ---: | ---: |
| Raw image dimensions unchanged | 64 | 64 |
| Folded page dimensions unchanged | 64 | 51 |
| All folding boundaries unchanged | 53 | 0 |
| Observer image dimensions unchanged | 64 | 51 |
| Reference remains an admissible source span | 64 | 0 |
| Cases with shifted retained ink | 11 | 64 |
| Cases with shifted retained reference ink | 0 | 0 |

Question erasure preserves every admissible answer image. Its 82,557 retained
reference ink pixels occupy the same folded coordinates. In 11 cases other
retained ink moves (4,730 pixels total); the audit does not claim complete
layout invariance or a uniquely semantic intervention.

Passage erasure removes every admissible reference in this cohort. Zero exact
answers are therefore forced by the source-copy output space; they cannot
independently demonstrate that the reader understood or needed the passage.
All 64 folding boundary sequences change, retained ink moves in every case,
and observer dimensions change in 13. There is no retained reference ink to
test for positional stability in this condition.

The running comparison and its registered decision rules remain unchanged.
Its normal-question accuracy, shared-passage question pairs, copying retention
and erasure outcomes must be reported together. A question-erasure difference
measures sensitivity to the actual transformed input, rather than proving a
unique internal mechanism. No language-quality improvement follows from
physical supervision validity or these geometry checks. The independent
free image-language objective remains unachieved.
