# Matched reader baseline: reading survives, image writing fails

The inherited visual reader answers 12 of 16 Chinese development questions
with literal reference strings through the same image path used by the failed
pixel-language model. It answers none literally when the question is erased.
Both date questions receive the correct years, `1874年` and `2001年`; the
learned pixel writer previously emitted damaged date-like forms. This is
evidence that useful input reading survives in the inherited body while its
new writing interface loses substantial answer capability.

| Input route | Cases | Literal reference matches | Natural stops | Caps |
| --- | ---: | ---: | ---: | ---: |
| Original question image | 16 | 12 | 16 | 0 |
| Question-erased image | 16 | 0 | 15 | 1 |
| Original source text | 16 | 11 | 16 | 0 |

The [complete report](evidence/pixel_language_source_reader/report.json)
retains all 48 decoded responses, including refusals and the capped erased
response. Some literal mismatches are aliases or longer relevant wording:
`Marilyn Manson` differs from the Chinese reference name, for example. These
are not independently graded semantic-accuracy estimates. All sixteen cases
are reused development examples, not a sealed test. No development response
or hidden state is permitted in training.

All 32 normal and erased prefills reproduce the actual ILM reader-memory
fingerprints exactly. The pretrained body, native image-budget transform,
numeric task frame and contextual computation are shared. Source adapter
outputs are zero, and the learned initial writing query and raster head are
absent. The [protocol](pixel_language_source_reader_protocol.md) specifies
the controls and the 64-unit response cap.

This is deliberately a **helper-only symbolic baseline**. Its temporary
source output projection and previous-symbol embedding lookup remain local
diagnostic variables, outside the image actor. It generates zero ILM image
answers and performs zero optimizer updates. Its success cannot substitute
for an independent image-output model. The matched comparison does not
uniquely separate failures in the learned writing query, image distribution,
and generated-image feedback.

The successful execution took 233.37 seconds, including 40.04 seconds waiting
for host memory pressure to recover, and peaked at 18.970 GiB GPU allocation.
Available RAM stayed above 71.66 GiB. Observed pressure averages reached
2.46% during paused samples; further reader computation waited for the
registered limits to recover. An earlier pressure-guard exit is retained
locally and is not included as a completed response. No other project's
process or kernel setting was changed. Original completed training sources
and admission rules remain unchanged.

The next intervention separates glyph formation from answer continuation.
A continuous image encoder and glyph decoder will be trained independently
of the question-answer readout. The proposed loop preserves the original
assistant state, predicts one complete glyph, and re-encodes the actual
generated pixels before continuation. Clear reconstruction alone will not
validate language: the repaired system must produce readable, correct,
autonomously terminated answers to unseen image prompts. This remains an
untested repair hypothesis.
