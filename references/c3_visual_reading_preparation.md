# Chinese reading cohort prepared before the new readout result

A separate image-only reading cohort is ready for later assessment. It contains
sixteen public C3 development passages and sixty-four input pages: original and
rotated options, each with an option-letter instruction and a full-answer-phrase
instruction. No model has been evaluated on it. The current genuine-copy cache,
training recipe and qualification remain unchanged.

Source files come from the authors' [C3 repository](https://github.com/nlpdata/c3)
at commit `1f41682ebb684d63afe7c583c8f38e137fc8a826`. The
[source notice](https://github.com/nlpdata/c3/blob/1f41682ebb684d63afe7c583c8f38e137fc8a826/license.txt)
specifies non-commercial research use. Raw texts, labels and rendered pages
remain in ignored data/artifacts directories with that notice. The public
[receipt](evidence/c3_visual_reading_preparation_v1/receipt.json) retains source
and input hashes, not the corpus text or reference labels.

Preparation took the first question from sixteen hash-ordered documents. None
of these sixteen required a font, layout or exact-CMRC-context-overlap rejection.
Fifteen passages fit at 24 pixels; the remaining passage fits at 20 pixels on the
fixed 960x1024 page. No answer-aware crop or truncation is used. All expected
phrases happen to be within the current canonical glyph coverage, with maximum
length 14; that coverage did not select the cases. This bounded cohort is not
historical-script, unknown-glyph or general language validation.

All sixteen four-view preview boards were actually inspected, as were full-size
phrase inputs for documents 0, 5, 10 and 15. Headers, complete passages, questions
and choices are separate and legible. The smallest-font full page is included
in that inspection. The [input review](evidence/c3_visual_reading_preparation_v1/input_review.json)
records the exact hashes and scope. It is an input-layout review, not a model
quality review or independent human grading.

The option rotation changes the correct letter while preserving the correct
phrase. Later results must report both-view consistency and complete phrase
writing separately, including stopping and all failures. Single-letter success
cannot replace full image-answer generation. The preparation predates genuine
readout qualification, and its source is absent from the present adaptation's
training bindings. Exposure in the inherited body's pretraining is unknown.

Local files:

| Artifact | Repository-relative path |
| --- | --- |
| Pinned development source and notice | `data/evaluation/c3_m_development_v1/` |
| 64 actual input PNGs | `artifacts/c3_visual_reading_v1/prompts/` |
| 16 paired preview boards | `artifacts/c3_visual_reading_v1/previews/` |
| Separate evaluator labels | `artifacts/c3_visual_reading_v1/evaluation_references.json` |
| Bound preparation and actual input review | `artifacts/c3_visual_reading_v1/` |

The [protocol](c3_visual_reading_protocol.md) and
[preparation code](../scripts/prepare_c3_visual_reading.py) specify acquisition,
selection and rendering. The current neural pipeline must finish first. This
preparation makes no new capability claim and supplies no training examples.

The [assessment executable](../scripts/assess_c3_glyph_language.py) validates
these reviewed inputs without reading evaluator labels:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/assess_c3_glyph_language.py --stage check-inputs
```

Its `prepare` stage requires the prospectively fixed genuine-copy candidate to
pass physical qualification and finish the original forty-stream assessment,
actual image review and scoring. It then freezes that candidate and the
preserved baseline. `generate` evaluates both through the image-only runtime
under the shared GPU mutex; `review` validates and displays all 128 raw streams.
`score` requires an actual output-first review before opening reference labels.
No C3 generation is started by preparation or by the current supervisor.
