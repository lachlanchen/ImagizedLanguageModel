# Sustained reader-query learning fails the paired-answer fit

The fixed 128-update experiment completes on one RTX 4090 D in 1,447.389 s,
including loading, training and generation, at 15.361 GiB peak allocation.
It still produces no exact seen or held-out answer. The two questions about
one seen passage therefore remain unfitted. This result does not support
scaling the unchanged query-only recipe.

| Canonical OCR-exact observation | Before | After |
| --- | ---: | ---: |
| Seen question pair | 0/2 | 0/2 |
| Excluded question pair | 0/2 | 0/2 |
| Seen/held Chinese and English copies | 0/4 | 0/4 |
| Seen questions, second generation seed | Not scheduled | 0/2 |
| Seen questions, question erased | Not scheduled | 0/2 |
| Seen questions, passage erased | Not scheduled | 0/2 |

Five registered answer/dependence gates fail. Copying retention passes only
because both copy counts are zero; it does not qualify writing. All eight
unique clean reference images pass the fixed OCR rule. These reused small
development cohorts are not a benchmark or a test of historical scripts.

The [protocol](native_query_fit_protocol.md) fixes 128 AdamW updates of the
shared 262,144 continuous reader-query coordinates. All other model weights
stay frozen, including the completed connector/writer adapters. The deployed
body retains 6,240,279,775 pretrained parameters. Sixteen updates use fully
independent noise; 112 use the other registered native noise levels. Inference
uses the unchanged fifty-step guidance-one image sampler. No transcript,
OCR training loss or changing word condition enters this learning path.

The [independent audit](evidence/native_query_fit_result/report.json) checks
all 128 schedule records, four real optimizer checkpoints, six complete saved
gradient/loss tensor sets and all 22 original output pages. Every FP32 query
coordinate changes, and 251,820 of 262,144 coordinates also change after
conversion to native BF16. The saved optimizer tensors remain finite and
their counters reach the prescribed steps. The first two update files are
byte-identical to the earlier two-update preflight; numerical replay passes.
The frozen-weight version checks pass during training.

For the two seen questions at the same registered full-noise input, velocity
MSE changes from 0.003543 to 0.003429 and from 0.022216 to 0.019156. The
normalized question-field alignment rises from 0.074015 to 0.146650. This is
an unbounded field measure on one passage, not accuracy. Its improvement
coexists with failed autonomous answers. The observations establish physical
optimization but do not identify a unique cause of the language failure.

All 18 new pages and four reused original baseline pages are authenticated
against actual decoder tensors and inspected at full resolution. The
[raw review](evidence/native_query_fit_result/raw_review.json) retains
malformed title-like fragments, copied question text, omitted forms and
deformed letters. Source and OCR access were available; no blinded or
quantitative manual accuracy is assigned. The question-erased pair's actual
decoded pixels match exactly. All eight reference pages are also inspected.
The original [inputs](evidence/native_query_fit_result/input_images/),
[outputs](evidence/native_query_fit_result/images/),
[references](evidence/native_query_fit_result/references/), unchanged
[OCR reports](evidence/native_query_fit_result/ocr/) and
[archive hashes](evidence/native_query_fit_result/provenance.json) remain
available with attribution.

The GPU child and coordinator terminate normally. No failed draw is rerolled,
and no optimizer or image-generation call is added for the audit. Current
evidence calls for testing the reader-to-writer conditioning interface before
another larger run. A supplied-answer teacher/motor control could distinguish
whether the existing image writer renders the intended answer when a correct
condition is available. Such a control would supply supervision and would
not itself demonstrate image-question understanding.
