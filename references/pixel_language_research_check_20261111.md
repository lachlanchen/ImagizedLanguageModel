# Research check: reading, writing and generated-history learning

Checked against primary sources on 11 November 2026 during the real-state cache.
This review does not alter that experiment's prepared inputs, split, fitted
parameters or thresholds. It identifies what the external work actually tests.

## Visual Chinese input is useful evidence, with a different output contract

Xiang and Guan's September revision, *One Image, No Tokens*, compares raster
and index inputs with a shared decoder. Its data section retains a 3,002-class
output vocabulary; its reported C-Eval procedure scores option letters rather
than producing free image answers. This supports studying visual input while
leaving our image-output and actual-feedback requirements unresolved. It is
not evidence that a Unicode-free complete image-language runtime has already
been demonstrated by that comparison.
[Primary paper, sections 3.2–3.4 and 6.3](https://arxiv.org/html/2607.03994v2).

## Multilingual pixel generation is established prior work

MIXAR v2 uses generated rollouts in GRPO refinement after extensive multilingual
pixel pretraining. Its stated free-generation results remain much weaker for
Chinese than its discriminative results. This is relevant precedent for
generated-history learning, not evidence of instruction-following parity with
a large text model or of cheap foundation training.
[Primary paper, section 4 and appendix K](https://arxiv.org/html/2604.11575v2).

The current 477M stage-2 author release has the **same weight SHA256, config,
all five Python files and rendering assets** as the already tested local release. The repository
revision differs, but this recheck supplies no new neural candidate to rerun.
No additional weights are downloaded. The card describes GRPO; the existing
local baseline must not be relabelled as an obsolete GAN checkpoint.
[Author release](https://huggingface.co/TYTTYTTYT/llama-pixar-477m-p32-stage2),
[local metadata/hash recheck](evidence/mixar_release_recheck_20261111.json),
[complete rendering/code recheck](evidence/mixar_rendering_release_recheck_20261111.json).

## Font synthesis does not choose a linguistic answer

GAR-Font conditions synthesis on a content glyph and style references, with
optional textual style guidance. It uses globally contextualized visual codes
and pixel supervision. Its stated task makes it relevant to a future style
renderer; it does not establish how to select the correct answer glyph from
a language question. Our known-form motor is already highly accurate under
the right condition, so replacing it with a font generator is not yet a
measured remedy for the readout failure.
[Primary method, sections 3.1–3.2](https://arxiv.org/html/2601.01593v1).

## Generated-state supervision needs the right expert

DAgger formalizes learning on the states induced by the learner, under its
imitation-learning assumptions. That distinction applies to our use of actual
generated pixels, but the theorem does not certify a proposed glyph-training
recipe. In particular, next-glyph supervision after a wrong prefix requires
an appropriate expert action or an explicit alignment objective. Pixel
reconstruction on reference histories and successful closed-loop language are
different tests.
[Primary proceedings article and paper](https://proceedings.mlr.press/v15/ross11a.html).

## Consequence for the active work

The [mathematical note](glyph_readout_and_clock_math.md) separates first-form
readout, physical image coherence and language-clock advancement. The active
natural-state experiment targets the first of these with real complete-image
prefills. Its subsequent qualification cannot prove the other two or the full
project objective. The current reader pass retains the disclosed pretrained
body; later efficiency claims must count this cache and all upstream costs.
