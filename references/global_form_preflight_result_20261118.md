# Whole-form supervision does not repair conditional drawing

The registered candidate completes all 96 whole-record updates and fails seven
of nine support checks. Physical F1 is .455463432, compared with the unchanged
tree control's .455979811. Neither produces an exact target mask; both put ink
at all four true empty positions. No trained successor or full language fit is
admitted. This rejects the tested fixed-budget recipe, not all auxiliary
supervision or image-based language learning.

| Complete training record | Positions | Saved tree F1 | Candidate F1 | Helper correct |
|---|---:|---:|---:|---:|
| complete_513_book | 512 | .455571809 | .455949038 | 66/512 |
| complete_qa | 5 | .378807947 | .383136095 | 4/5 |
| longest_native_record | 35 | .469458792 | .452796685 | 9/35 |
| Pooled | 552 | .455979811 | .455463432 | 79/552 |

The temporary helper's source-form accuracy rises from zero to .143115942.
Its cross-entropy falls on every source, but both pooled and complete-case
accuracy requirements fail. Even the short question's four correctly classified
positions have no exact emitted mask. The helper is absent from generation:
its predicted class never selects an image or enters the physical prefix.
Classification and actual writing are different computations. This result does
not identify a unique cause, show semantic understanding or qualify readability.

The physical writer's source NLL falls on all three records. Final values are
213.412, 101.737 and 184.754 nats/image. Pooled rotated-minus-correct context NLL
is .395939 nat/image, below the fixed .5 requirement and the control's .707847.
It passes only the retained gain over the much worse flat head and zero false
blanks. A nonempty damaged image can satisfy the latter condition. No threshold
is changed and no intermediate checkpoint is selected.

## Execution and complete evidence

Five CPU implementation checks pass. Source-only preparation constructs 9,581
classes from exact physical masks, with exact coverage of all 552 targets.
The candidate uses the same fresh physical tensors, complete source order,
96-update schedule and greedy generator as the saved control. All initial
images are bit identical to that control. No old control is re-executed.

All three complete-record probes have finite positive auxiliary gradients in
the actual shared condition, physical position and span-length embeddings;
the helper alone supplies no gradient to physical-prefix blocks. Joint gradients
and context gradients are finite. Every actual generated event passes cached/
parallel probability replay within 1.073e-6 nat. All physical and helper trial
tensors reset exactly; original actor weights, cache bytes and prior language
readings remain unchanged. No fitted candidate is installed.

The [CPU exporter](../scripts/export_global_form_preflight.py) independently
recounts every one of the 1,104 actual initial/final images, decodes every
retained physical trace and checks every helper prediction against its saved
source label. All six [lossless image sheets](evidence/global_form_preflight_20261118/)
preserve the emitted pixels. The persisted cached/parallel probabilities are
recounted without another neural call; NLL and CE remain measurements of the
original bound execution. The complete [hash-bound result](evidence/global_form_preflight_result_20261118.json)
retains all cases, 96 updates, raw-file hashes, gradients and failed conditions.

The candidate takes 59.973 monitored seconds, zero pressure wait and .233 GiB
peak allocation, with 52.156 GiB minimum sampled available host memory. The
outer supervisor records 70.071 seconds and exits successfully. GPU0 is released
and no project GUI remains. Foreign GPU1 is unchanged. There are 2,261,074
physical-writer parameters and 1,886,189 training-only helper parameters. These
are head-only costs: the reused 1,465,381,142-parameter source actor, inherited
pretraining and earlier cache preparation are excluded from this execution but
remain required provenance. This is not a whole-model efficiency comparison.

The experiment uses authentic training records with correct supplied histories
and span lengths; glyphs within a span remain conditionally independent. It
does not measure held-out, autonomous, historical-form or complete-language
quality. Preserve the limited classification improvement and the failed drawing
together. Do not extend unchanged epochs, relax conditions or call the helper
a language model. The full engineering goal remains active.
