# Literal writing transfer remains unsuccessful

The completed full-corpus clean checkpoint reproduces
**0/12 requested answer images exactly**
on the fixed Chinese, English and six-digit copying requests. Overall
foreground F1 is 22.486%. Learned stopping
occurs on 12/12 outputs. Both answers are exact
in 0/6 pairs.

| Requested writing | Exact images | Foreground F1 |
| --- | ---: | ---: |
| all | 0/12 | 22.486% |
| zh | 0/4 | 26.989% |
| en | 0/4 | 15.925% |
| numeric | 0/4 | 24.543% |

These are 12 descriptive transfer cases with exact-prompt-disjoint training
images, not a general benchmark or a proof that visual language learning is
impossible. They show that this checkpoint does not reliably preserve writing
requested in new prompts. Mean owning-target minus partner-target F1 is
0.123 points. This comparison is offline
cross scoring, not an additional set of model calls.

Inference receives only the actual rendered prompt image. References are
loaded for scoring after all model calls; no target length, code, OCR,
character identifier or teacher service enters generation. Independent CPU
verification authenticates all 12 raw outputs, reconstructs source layout,
checks stopping events and recomputes physical scores. It does not replay
the entire neural generation or every feedback encoding.

Every prepared input, reference and nonempty raw output PNG is copied under
evidence/copy_transfer_v75_raw/. No output is repaired, cropped or selected
for appearance. The [receipt](evidence/copy_transfer_v75_result.json) binds
every exported file and the [protocol](copy_transfer_v75_protocol.md).
The full original 192-question audit is reported separately. The general
usable ILM goal remains unachieved.
