# Native visual flow improves a few copies but fails question answering

The complete fixed initial/final assessment produces no correct Chinese
question answers in either arm. Exact copying rises from 0/64 to 5/64 after
1,536 adapter updates. All 384 original answer images have been authenticated
against their saved decoder tensors and visually inspected. The candidate
has not demonstrated useful language learning from glyphs.

| Fixed OCR-exact outcome | Initial/source | Final/trained |
| --- | ---: | ---: |
| Seen questions, any source-accepted answer | 0/32 | 0/32 |
| Article-disjoint development questions, any source-accepted answer | 0/32 | 0/32 |
| Both answers correct for a held-out passage | 0/16 | 0/16 |
| Question-erased held-out questions | 0/32 | 0/32 |
| Passage-erased held-out questions | 0/32 | 0/32 |
| Seen copying strings | 0/32 | 2/32 |
| Held-out copying strings | 0/32 | 3/32 |

The [complete numerical report](evidence/native_visual_flow_complete_result/assessment.json)
contains every observation and the unchanged developmental gates. Seen fit,
held-out answers, held-out pairs, question dependence and passage dependence
all fail. Copying retention passes against the initial arm's zero count;
that relative criterion does not qualify reliable writing. Erasing the
question makes both inputs and their seeds identical within each pair, and
the actual decoded outputs then match bit-exactly in both arms.

Clean-reference OCR is imperfect: 23/32 seen and 26/32 held-out QA images,
and 27/32 seen and 29/32 held-out copying images, match their reference strings.
No uncertain output was manually repaired or rescored. The
[complete review record](evidence/native_visual_flow_complete_result/raw_review.json)
binds all 384 actual page inspections. Review had source and OCR access and
does not supply independent blinding or a manual accuracy estimate. The
assessment's original `raw_review_completed: false` records its state when
numerical collection finished; the separate complete review record documents
the subsequent inspection without rewriting that report.

The initial arm retains the same pretrained reader and image generator with
zero-output adapters. The final arm loads the fixed completed adapter export.
Both use the original image inputs, seeds, compiled task, 1024-square canvas,
50-step sampler and guidance 4. External pretraining remains part of the
6,240,279,775-parameter runtime; this is not a model trained from scratch.
The article-disjoint development cases have been reused across experiments.
Neither these results nor their runtime establish efficiency at useful quality.

## Conditional-field diagnostic

The registered [18-case field diagnostic](native_flow_condition_probe_protocol.md)
has now completed all 144 forwards. At full noise, the actual noisy input is
exactly independent noise; the reference answer supplies offline supervision
only. Predicted fields swap bit-exactly under question swaps, and the erased
control gives exact zero alignment. The
[actual per-pair results](evidence/native_visual_flow_complete_result/condition_probe.json)
retain positive and negative observations.

| Mean normalized alignment, four normal pairs per split | Initial/source | Final/trained |
| --- | ---: | ---: |
| Seen pairs | 0.006717 | 0.010039 |
| Held-out pairs | 0.003767 | 0.009684 |

This unbounded continuous statistic is not answer accuracy or a percentage.
Perfect target-field matching gives one, question independence gives zero,
and exactly swapped matching gives minus one. The observed small positive
means show a limited conditional signal in this diagnostic, alongside negative
individual pairs. Four pairs per split, inherited pretraining, answer layout
and partial lexical overlap do not establish semantic acquisition or a unique
cause of the failed images.

## Sampling the branch used in training

The registered [guidance-1 diagnostic](native_flow_positive_sampling_protocol.md)
also completes: four original questions per arm produce zero correct answers
at guidance 1 and zero at guidance 4. Both newly executed guidance-4 replays
match their original native fields, decoded tensors and PNG pixels exactly.
The [numerical result](evidence/native_visual_flow_complete_result/positive_sampling_probe.json)
and [all ten original diagnostic pages](evidence/native_visual_flow_complete_result/sampling_raw_review.json)
retain every attempt. All ten pages were actually inspected. These four reused
failing cases are a small diagnostic, not a new benchmark. Changing guidance
alone does not repair them and does not exclude other sampling effects.

## The saved codec reconstructions preserve the tested glyphs

The completed preflight already saved four answer reconstructions using both
posterior mode and a fixed posterior sample. A new CPU audit authenticates
all eight original images and reproduces every ink measurement. The fixed OCR
evaluator reads all four clean targets and all eight reconstructions exactly;
foreground pixel F1 ranges from 0.966741 to 0.988286. All eight original pages
were inspected. The [codec audit and images](evidence/native_visual_flow_complete_result/codec/report.json)
are available with their [inspection bindings](evidence/native_visual_flow_complete_result/codec/raw_review.json).
No new codec, generator or optimizer call is performed by this audit.

This is reconstruction with the correct target already encoded, not autonomous
answer generation. It demonstrates representability for four short seen
answers at this geometry, including Chinese and English copying. It does not
qualify every glyph or historical writing system. Together with failed sampled
answers, it places the next learning question in the conditional reader/writer
path for these cases rather than an inability of this codec to represent them.
It does not isolate which learned component needs repair.

Reproduce the saved-image audit with the existing model Python and a fresh
output folder:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/audit_native_flow_codec_reconstructions.py \
  --output artifacts/native_flow_codec_reconstruction_audit_reproduction
```

## Assessment execution and provenance

The original aggregation failed on a Python name collision after all images
and OCR reports completed. The [bounded recovery](native_flow_assessment_recovery.md)
aliases only the imported schedule-digest function and its call, preserving
all 19 registered source files. The corrected CPU execution authenticates
all 384 existing draws and applies the original scores and gates. Its
[provenance](evidence/native_visual_flow_complete_result/provenance.json)
records both execution-source hashes, unchanged precursor stages and every
exported result/image hash. No assessment image was regenerated.

The earlier complete [initial/final normal evidence](native_visual_flow_normal_comparison.md)
and [final copying evidence](native_visual_flow_copy_result.md) retain their
original full pages. QA source attribution is recorded with the
[CMRC evidence](evidence/native_visual_flow_final_normal/SOURCE_ATTRIBUTION.md);
synthetic copying attribution is [separate](evidence/native_visual_flow_final_copy/SOURCE_ATTRIBUTION.md).
Correct autonomous unseen answers, reliable writing and historical-form
semantics remain requirements for the independent ILM objective.
