# Native flow training gives no exact-answer gain on the normal cohort

The complete matched normal-question comparison yields zero exact answers
before and after the 1,536-update adapter training run. This result holds for
both the canonical reference and all source-accepted answer strings. Both
questions are correct in zero same-passage pairs in either split or arm.

| Normal-question metric | Initial seen | Final seen | Initial held out | Final held out |
| --- | ---: | ---: | ---: | ---: |
| Canonical OCR-exact answers | 0/32 | 0/32 | 0/32 | 0/32 |
| Source-accepted OCR-exact answers | 0/32 | 0/32 | 0/32 | 0/32 |
| Both answers correct for a passage | 0/16 | 0/16 | 0/16 | 0/16 |

The [matched comparison record](evidence/native_visual_flow_normal_comparison.json)
joins all 64 cases by their original order and verifies identical IDs, split,
passage pair, input hashes, seeds, reference-image bytes and source-accepted
answer sets. The deployment, source audit and owning OCR calibration also
match. The initial arm uses the retained pretrained source with zero-output
adapters; the final arm loads the fixed final adapter export. Both use the
registered 50-step, guidance-4 native image sampler. This is not a comparison
against a randomly initialized language model.

The complete [initial evidence](evidence/native_visual_flow_initial_normal/report.json)
and [final evidence](evidence/native_visual_flow_final_normal/report.json)
retain every original input, generated answer and reference image. All 128
normal-question output pages were visually inspected and authenticated
against their saved actual decoder tensors. The initial pages often contain
book-like headings and paragraphs of malformed writing; the final pages
contain short fragments with malformed or merged glyphs. Output appearance
changes without a gain in the registered exact-answer metric. No manual
accuracy, complete manual transcription or OCR correction is assigned.

OCR of clean reference pages matches 23/32 seen and 26/32 held-out answers.
These imperfect calibration counts limit the automated metric. Inspection
had source and OCR access, so it is not an independent blinded evaluation.
The held-out articles never enter this optimizer schedule, but their
development questions have been reused across experiments.

| First held-out case | Complete input | Initial answer | Final answer | Reference |
| --- | --- | --- | --- | --- |
| Question 1 | [Input](evidence/native_visual_flow_initial_normal/images/032_input.png) | [Initial](evidence/native_visual_flow_initial_normal/images/032_answer.png) | [Final](evidence/native_visual_flow_final_normal/images/032_answer.png) | [Reference](evidence/native_visual_flow_initial_normal/images/032_reference.png) |
| Question 2 | [Input](evidence/native_visual_flow_initial_normal/images/033_input.png) | [Initial](evidence/native_visual_flow_initial_normal/images/033_answer.png) | [Final](evidence/native_visual_flow_final_normal/images/033_answer.png) | [Reference](evidence/native_visual_flow_initial_normal/images/033_reference.png) |

The complete erasure/copying comparison remains in progress. This completed
normal cohort supplies no causal explanation for the failure. The already
registered [conditional-field probe](native_flow_condition_probe_protocol.md)
and [positive-branch sampling diagnostic](native_flow_positive_sampling_protocol.md)
remain separate pending observations. Useful language learning from glyphs,
historical-script semantics and efficiency at matched answer quality remain
unverified.

Reproduce the initial export with the unchanged audit used for the final arm:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/audit_native_flow_normal_cohort.py \
  --assessment artifacts/native_visual_flow_assessment_v1 \
  --review artifacts/native_flow_initial_normal_review_v1 \
  --arm initial --output artifacts/native_flow_initial_normal_audit_reproduction
```

Use the existing model Python and a fresh output directory. The audit requires
the complete actual raw-inspection record and authenticates saved tensors on
CPU. It generates no model answers and leaves the frozen full assessor intact.
