# Full native pixel-language adaptation: V61 result

The complete native image-language model trains on one RTX 4090 D, but the
registered language qualification fails. All 477,145,984 parameters make
1,000 updates, eight accumulated image examples per update, in 1,501.138 s
through the final optimizer update. Peak allocated memory is 9.912 GiB.
Loading takes 2.525 s; preparation, external pretraining and final serialization
are excluded from that training time. The existing pinned MIXAR foundation
is external work, not single-GPU pretraining performed here.

| Fixed 256 development questions | Trained | Question blanked | Unadapted base |
| --- | ---: | ---: | ---: |
| Exact stopped image answers | 0 | 0 | 0 |
| Exact-raster multiset F1 | 0.149% | 0% | 0% |
| Aligned foreground pixel F1 | 12.264% | 8.473% | 0% |
| Mean output cells | 3.047 | 2.805 | 0 |
| Capped outputs | 0% | 0% | 0% |

All primary outputs stop, but the trained rasters are damaged fragments, not
usable written answers. No shared-paragraph pair has both answers exact
(0/393). Five of ten registered gates pass: image boundary, split/resources,
full backward/resume/causality, offline export replay and cap rate. Accuracy,
raster F1, adaptation gain, question-dependent raster F1 and paired correctness
fail. The decision is **reject_native_free_reading**.

Across all 1,321 development questions, answer/termination-pixel NLL improves
from 0.684156 to 0.154814 nats per pixel per example. This is not image NLL or
text perplexity. Foreground F1 has a 3.791-point normal-minus-blank difference
[2.521, 5.056], but ink dependence is not meaningful language. The exact-raster
F1 difference has a zero lower confidence bound. Article bootstrap uses 2,000
draws, seed 20266219. Development is explicitly reused, not a sealed final test.
The full audit takes 145.690 s excluding loading, at 1.832 GiB allocated memory.

The model receives only binary prompt images and writes new 32-by-64 pixel
blocks. No source-image selector, glyph inventory, Unicode IDs, tokenizer,
OCR or external model call enters inference. The native model's independent
pixel law and the Noto 26-pixel rendering shift are both plausible contributors
to failure; this run does not isolate either as the sole cause.

The complete backward preflight has finite gradients in every parameter.
Interrupted/resumed weights, EMA, optimizer and all recorded RNG states match
exactly. Flipping future answer pixels changes earlier predictions by zero.
The exported learned bundle includes its model code and weights and replays
tensors, PNG bytes and events exactly with empty source/model caches and no
network access. These establish implementation and deployment, not language
quality.

The separately preregistered answer-only Tesseract diagnostic finds zero exact
trained transcripts. It recognizes only 40/256 first-reference images exactly
(15.625%; bounded character accuracy 35.532%), so its semantic measurements
are poorly calibrated for this visual format. No question or reference text
is supplied to the recognizer. Its result does not change the primary decision.

Data, optimizer/model states, deployment and raw outputs are under
`artifacts/native_pixel_instruction_v61_20261105/`: `data/`, `training/`,
`standalone/`, `development/` and `transcript_diagnostic/`.
See the [frozen protocol](native_pixel_instruction_v61_protocol.md),
[full code guide](../docs/native-pixel-language.md) and
[physics/mathematics review](physics_math_routes_20261105.md).
