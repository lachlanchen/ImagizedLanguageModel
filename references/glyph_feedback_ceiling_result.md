# Native language steps and glyph steps are not interchangeable

The matched helper comparison is complete. Replacing the source model's native
steps with first-character steps sharply reduces literal answer matching, even
with exact source character input vectors. The learned image encoder is not
the only obstacle. This is an offline diagnostic, not a generated-image result.

| Route | Training controls exact /8 | Development exact /16 | Development natural terminals | Unsupported actions | Caps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native units and exact native feedback | 5 | 12 | 16 | 0 | 0 |
| First character and exact character feedback | 5 | 3 | 9 | 5 | 2 |
| First character and encoded canonical-image feedback | 5 | 3 | 10 | 4 | 2 |

All 72 initial states match the saved image-model states exactly, with identical
geometry. The body, input image and complete native decision head are held fixed.
The native-unit row independently reproduces the previous twelve-of-sixteen
Chinese source-reader literal result. All three routes return `288条`, `1874年`
and `2001年` on the same three development cases. Other differences include:

- The native response `批评他的人` becomes a capped repetition of `的` in both
  character-feedback routes.
- The native `中国浙江省嘉兴平湖市` becomes repeated or omitted characters with
  exact character vectors, and a longer damaged location description with
  canonical-image feedback.
- Native `85分钟` becomes `85分` under both character routes. The image writer's
  incomplete output therefore has a corresponding failure in this helper.

The [full records](evidence/glyph_feedback_ceiling_v1/scores.json) retain all
outputs, including alternate wording, wrong answers, partial Unicode failures
and caps. Literal reference matching undercounts semantically equivalent longer
answers. No post-hoc semantic accuracy or newly generated ILM image is assigned.
The helper uses its complete source head, tokenizer and canonical image bank;
those tools are absent from the learned image runtime. Diagnostic records are
explicitly excluded from subsequent training, including the training controls.

Formally, if the native winner is a source unit \(u\) with embedding \(w_u\),
the tested character policy substitutes
\(w_{\mathrm{first}(u)}\) or \(E(I_{\mathrm{first}(u)})\) for \(w_u\), then
advances the contextual body one step. A multi-character unit is not consumed
by that substitution. Repeated first-character selection, changed positions
and partial Unicode units can therefore alter the trajectory before any learned
image-writing error occurs. The exact-vector counterfactual shows that reducing
image-encoding error alone cannot preserve this source policy.

This does not prove that the frozen state contains insufficient information
for a separately trained character decoder. Nor does equal three-case matching
prove that encoder error is harmless: individual trajectories differ. It does
show that the native source head cannot be treated as a valid next-glyph oracle
by simply taking the first character of each newly selected native unit.

The next candidate will train the final contextual attention layers jointly
with the pixel writer on complete glyph-feedback sequences. The frozen lower
layers can be cached once, so this tests an actual adaptation of the sequence
dynamics without backpropagating through the entire large body. Broader training
questions are needed because the original 508 histories are already fitted.
This is a proposed repair, not an achieved language result.

The diagnostic takes 353.62 seconds, including 40.04 seconds of pressure waiting,
at 20.041 GiB peak allocation; minimum available host RAM is 49.47 GiB. It makes
no optimizer update. The [prospective protocol](glyph_feedback_ceiling_protocol.md)
and all authenticated records preserve the tested limits and qualifications.
