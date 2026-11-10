# Fresh reading questions still expose a phrase-writing failure

The prospective C3 assessment is complete for both frozen checkpoints. The
pixel-only readout stops more often on letter instructions and produces some
clearer Latin forms. Neither checkpoint produces a correct complete phrase
under the fixed canonical comparison or the output-first readings. This result
does not qualify either model as a useful general image-language model.

| Checkpoint / requested output | Streams | Exact canonical images | Clear complete transcription matches | Natural stops | Caps | Both option orders clear and correct /16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Preserved baseline / letter | 32 | 0 | 6 | 19 | 13 | 1 |
| Preserved baseline / full phrase | 32 | 0 | 0 | 4 | 28 | 0 |
| Pixel-only G / letter | 32 | 0 | 11 | 27 | 5 | 3 |
| Pixel-only G / full phrase | 32 | 0 | 0 | 8 | 24 | 0 |

All sixteen source documents have original and rotated option orders, and
both letter and full-phrase instructions. The passage, question and complete
options appear in each input image. The two checkpoints were fixed before any
C3 model call. There is no fitting or checkpoint selection from these results.
The [prospective preparation](c3_visual_reading_preparation.md) retains the
source pin, input review and selection rule. Inherited-body pretraining exposure
is unknown; this is fresh to the current adaptation, not a proved clean-room
benchmark. The options have two, three or four choices, so a letter count alone
is especially inadequate as evidence of general reading competence.

All 128 raw output streams were actually inspected on sixteen boards before
opening the current reference answers. Five additional boards enlarge the 56
short naturally stopped outputs by an integer factor of three without smoothing
or stroke repair. Many A-like forms lack a crossbar, and some rounded forms
are ambiguous between B and 8. Those uncertain forms receive no confident
transcription. Readable wrong letters, including G, are retained as read rather
than snapped to the available options. The clear-reading counts are conservative
agent observations, not an independent human accuracy estimate. Canonical
pixel comparisons and these readings remain separate.

Most requested phrases become damaged repetition. Some stopped outputs contain
only a letter or malformed short string; a natural stop is not a correct answer.
The [complete evidence archive](evidence/c3_glyph_language_assessment_v1/archive.json)
retains every generated output PNG, all viewed boards, per-stream measurements,
stop/cap decisions and uncertainty. Inputs and source reference strings remain
local; public per-stream references use hashes. The source is restricted to
non-commercial research under the preserved C3 source notice. This repository's
code license does not replace that data restriction.

Generation takes 1,022.06 seconds for both checkpoints together, including
loading and resource guards; the full resource receipt is archived. The runtime
uses the inherited 7.5B visual/contextual body and actual emitted-pixel feedback,
with no tokenizer, source output head, font, glyph lookup or teacher service.
These timings are not from-scratch language training or a matched-quality
efficiency advantage. The failed phrase result reinforces the need for the
[matched feedback diagnosis](glyph_feedback_ceiling_protocol.md) before another
readout training change.
