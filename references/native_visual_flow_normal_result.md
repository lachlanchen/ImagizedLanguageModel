# The trained image-answer model fails the complete normal-question cohort

After 1,536 optimizer updates, the final native visual-flow model obtains
zero OCR-exact answers on both the 32 seen questions and the 32 article-disjoint
development questions. Matching against every source-provided acceptable answer
leaves both counts at zero. The completed
[matched initial/source normal comparison](native_visual_flow_normal_comparison.md)
also gives zero exact answers in both arms. The complete erasure/copying
comparison remains in progress, and a causal explanation is not established.

The completed [copying cohort](native_visual_flow_copy_result.md) also fails
to establish reliable writing: 2/32 seen strings and 3/32 held-out strings
match exactly under the registered OCR rule. All 64 copying outputs have
been inspected and authenticated against their actual decoder tensors.

| Completed final normal cohort | Seen | Held out |
| --- | ---: | ---: |
| Questions | 32 | 32 |
| Canonical OCR-exact answers | 0 | 0 |
| Source-accepted OCR-exact answers | 0 | 0 |
| Both answers correct for the same passage | 0/16 | 0/16 |
| OCR-exact clean reference images | 23/32 | 26/32 |

The [complete evidence](evidence/native_visual_flow_final_normal/report.json)
retains all 64 original input pages, generated answers and padded reference
images. Every generated PNG authenticates against its saved actual decoder
tensor. Input order, seeds, deployment, source code, source-accepted answers,
normalization and reference calibration are checked against their original
bindings. No neural model, new image generation or transcript repair is used
by this offline audit.

All 64 generated pages were also visually inspected using all sixteen contact
sheets, with each complete 1024-square answer occupying its original pixels.
The [inspection record](evidence/native_visual_flow_final_normal/raw_inspection.json)
preserves per-page observations and image hashes. Many glyphs have malformed
or merged strokes, and several outputs contain very similar short fragments.
Some isolated characters and an English word are recognizable. The reviewer
had source-key, reference and OCR access; this is neither an independent
blinded assessment nor a quantitative manual-accuracy score. Uncertain
glyphs remain untranscribed. OCR calibration is imperfect and the automated
counts must retain that limitation.

These are the first same-passage pairs in each registered split, shown with
their full input, unmodified output and reference pages. The complete evidence
contains every remaining case, including failures.

| Case | Complete input | Generated answer | Reference |
| --- | --- | --- | --- |
| Seen, question 1 | [Input](evidence/native_visual_flow_final_normal/images/000_input.png) | [Answer](evidence/native_visual_flow_final_normal/images/000_answer.png) | [Reference](evidence/native_visual_flow_final_normal/images/000_reference.png) |
| Seen, question 2 | [Input](evidence/native_visual_flow_final_normal/images/001_input.png) | [Answer](evidence/native_visual_flow_final_normal/images/001_answer.png) | [Reference](evidence/native_visual_flow_final_normal/images/001_reference.png) |
| Held out, question 1 | [Input](evidence/native_visual_flow_final_normal/images/032_input.png) | [Answer](evidence/native_visual_flow_final_normal/images/032_answer.png) | [Reference](evidence/native_visual_flow_final_normal/images/032_reference.png) |
| Held out, question 2 | [Input](evidence/native_visual_flow_final_normal/images/033_input.png) | [Answer](evidence/native_visual_flow_final_normal/images/033_answer.png) | [Reference](evidence/native_visual_flow_final_normal/images/033_reference.png) |

The run adapts 37,380,992 parameters inside a retained pretrained system with
6,240,279,775 total runtime parameters. Its generator receives the changing
question/passage only as image pixels, with a compiled generic task condition.
Its deployment uses no changing textual answer, external answering teacher,
OCR engine or font lookup. The independent offline evaluator uses OCR after
generation. This information boundary and the completed one-card execution
do not establish useful language learning from glyphs. Historical-script
understanding, general English answering and efficiency at matched answer
quality remain unverified. The development questions have been reused across
experiments and are not a new untouched test set.

The [complete assessment](native_visual_flow_assessment_protocol.md) and
[registered condition diagnostic](native_flow_condition_probe_protocol.md)
continue unchanged. The latter compares the two questions at shared independent
noise, testing whether the predicted field moves toward the corresponding
answer. It separates a conditional-field observation from readable autonomous
sampling. More unchanged training is not justified by this failed cohort alone.

A separate [positive-branch sampling diagnostic](native_flow_positive_sampling_protocol.md)
tests the mismatch between positive-only optimization and guidance-4 inference.
It fixes the first passage pair in each split, requires exact native execution
replay, and changes only guidance to one. Its
[actual CPU preparation](evidence/native_flow_positive_sampling_preparation.json)
passes image bindings and refuses premature neural execution. The lightweight
coordinator waits behind the complete assessment and field probe. No new
sampling or answer-quality observation follows from this preparation.

Reproduce the complete-cohort audit after recording actual raw inspection:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/audit_native_flow_normal_cohort.py \
  --assessment artifacts/native_visual_flow_assessment_v1 \
  --review artifacts/native_flow_final_normal_review_v1 \
  --output artifacts/native_flow_final_normal_audit_v1
```

Use the existing model Python and a fresh output directory. The tool requires
the completed original normal cohort and owning calibration. It refuses
incomplete or changed bindings and preserves all input, answer and reference
PNG bytes. The full 384-generation assessor remains a separate frozen program.
Source data attribution and the CMRC2018 licence accompany the exported images.
