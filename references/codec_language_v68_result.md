# V68: question dependence without accurate answer writing

The fixed 10,000-update image-language actor fails its autonomous Chinese
development requirements. All 384 normal/control generations are complete.
Independent verification reproduces every physical score, checks all 192
original first references and reconstructs all 384 input layouts. The
[verification receipt](evidence/codec_language_v68_autonomous_result.json)
binds the 768 raw tensor/PNG files. The training, evaluator, data, references
and quality gates remain unchanged.

| Chinese development requirement | Measured | Required | Outcome |
| --- | ---: | ---: | --- |
| Full-image exact answers | 0/128 | At least 50% | Fail |
| Foreground F1 | 32.6713% | At least 80% | Fail |
| Both questions answered exactly | 0/64 pairs | At least 35% | Fail |
| Normal minus question-blank F1 | 13.1628 percentage points | At least 10 points | Pass |
| Article-bootstrap interval lower bound | 10.9329 points | Greater than zero | Pass |
| Learned stopping | 128/128 | At least 90% | Pass |

Blank-question Chinese F1 is 19.5085%. The 95% article-cluster bootstrap
interval for the difference is [10.9329, 15.7296] percentage points, with 64
articles, 2,000 resamples and the original seed. A resolved effect of question
pixels is insufficient for correct conditional writing. Stopping is also
insufficient: all primary outputs stop, while none reproduces a full answer.
The first fixed audit example, `DEV_477_QUERY_0`, contains fragmented glyphs.
No OCR, image replacement or output repair was applied.

The 64 English instructions have 9.7630% normal foreground F1 versus 3.9632%
with a blank prompt. Recorded-reference exact agreement is zero in both
routes. Normal learned stopping is 54/64. English figures measure agreement
with recorded images; they are not an exhaustive semantic assessment of
alternative valid answers. This run does not establish useful English writing.

## Execution and evidence

The [completed adaptation](codec_language_v68_training_result.md) consumes all
12,976 admitted training examples and records 20,000 draws. Final trainable
weights are fixed at SHA-256
`39d978ae7eefa4fff5a73084e7dd5b262ceb09ecd09b253e2d3627fe03a4b66c`.
External foundation and codec pretraining remain part of the provenance.

The audit uses 128 Chinese and 64 English prompts, each with its matched blank
control, a common 128-block output cap and learned stopping. It receives prompt
pixels without answer length, reference codes, source strings or case metadata.
Reused development is disclosed; no genuinely sealed final examples are opened.
The [control-layout correction](codec_language_v68_layout_control_correction.md)
was fixed before generation and preserves every normal input pixel.

Generation takes a measured 783.228 seconds summed over all streams. Audit
setup is 65.599 seconds and total evaluation time is 851.764 seconds. Peak
allocated inference memory is 15.685 GiB. These costs exclude prior training
and do not establish an efficiency advantage over a text model. The evaluator,
independent verifier and their coordinator have exited; GPU 0 is released.

Local evidence is under `artifacts/codec_language_v68_20261105/development/`:

- Audit-report SHA-256: `0b63ee86c8ee3c4053a5d8e4b1afdf0644ba9b5c9e82c4d8d8c6e5a955c44191`.
- Verification-receipt SHA-256: `ca09f2fb88374406cc2ddfc8e44637776ddbea2fc5c18240ebce83b8df9c026c`.
- Prepared-data SHA-256: `4f90406b51670e4e82327d1396e1c954e62bfb5d7b31c437b5c59e88a9180fe5`.

## Consequence for the next experiment

The supplied-code codec result remains a separate reconstruction result.
Correct code decoding does not imply that the contextual actor predicts those
codes. The current audit also cannot isolate failures present on exact answer
history from errors accumulated after generated blocks.

The next diagnostic therefore compares supplied-code reconstruction,
next-block prediction under exact reference history and autonomous generation,
using the same frozen final actor. Include training examples to test fitting
as well as reused development to test transfer. Reference-history prediction
is an explicitly assisted diagnostic, not autonomous language performance.
Select a learning intervention only after this distinction is measured.
Standalone export remains gated, and the general image-language goal remains
unachieved.
