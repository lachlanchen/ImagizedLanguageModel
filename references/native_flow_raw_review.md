# Review complete native pages without changing their pixels

`scripts/prepare_native_flow_review.py` packages complete 1024-square model
answers for visual inspection. It authenticates every original PNG against
its actual decoder tensor, copies the PNG bytes verbatim and places four
complete pages on each contact sheet. Each answer occupies its original
1024-by-1024 pixels. Only external review labels are added. Nothing is
resized, cropped, contrast-adjusted or transcribed.

The packet contains opaque review IDs, complete image files, contact sheets
and unfilled annotation records. Group names, source IDs, seeds and source
bindings stay in a separate key. Expected answers are never loaded by this
program. The preparer retains source access, so the packet does not establish
independent or perfect blinding. Legibility inspection supplies no manual
answer-accuracy score or replacement for the registered OCR assessment.

## Completed preparation check

The tool has packaged all 24 actual fixed training-monitoring pages into
six contact sheets. Every copied PNG and every embedded answer region is
verified against its source. The [receipt](evidence/native_flow_monitoring_review_preparation.json)
binds all files. The first and last contact sheets were inspected for layout:
labels remain outside image regions, complete pages fit, and the nearly blank
output remains present. This is a layout check, not a new review of all six
contact sheets or a new model result.

An actual assessment-mode invocation during an incomplete normal cohort
correctly refuses to create a review packet. Assessment mode requires the
complete original 64-case normal/copying cohort or 32-case erased cohort,
with its completed native report and exact original input/seed order.
It cannot silently package a partially generated cohort as complete.

Training-monitoring and assessment modes remain explicit. The training
previews are seen examples from different updates; their appearance cannot
select a checkpoint or establish unseen-question answering. Earlier direct
inspections of partial assessment outputs also remain separate from a
complete cohort review.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/prepare_native_flow_review.py \
  --mode monitoring --training-run artifacts/native_visual_flow_learning_v1 \
  --audit artifacts/native_visual_flow_training_audit_v1 \
  --output artifacts/native_flow_monitoring_review_v1

# After the original final normal cohort has completed:
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/prepare_native_flow_review.py \
  --mode assessment --assessment artifacts/native_visual_flow_assessment_v1 \
  --cohort final_normal --output artifacts/native_flow_final_normal_review_v1
```

Use the existing model Python, CUDA hidden and a fresh output directory.
Repeat `--cohort` to combine complete cohorts. Existing packets and source
images are never overwritten. Review completion must be recorded separately,
bound to the actual packet hashes, without inferring uncertain glyphs or
silently repairing scores. No neural model is loaded during preparation.

## Completed final normal-cohort inspection

Assessment mode has subsequently packaged the complete final normal cohort:
64 original answer pages in sixteen contact sheets. Every sheet and every
embedded answer page has now been visually inspected. The
[cohort result](native_visual_flow_normal_result.md) retains the complete
input/answer/reference evidence and per-page inspection bindings. This review
has source and OCR access and assigns no manual answer-accuracy score.
The remaining source, erasure and copying cohorts still require their own
complete assessment and raw review.
