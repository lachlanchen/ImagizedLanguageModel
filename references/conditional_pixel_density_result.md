# Direct pixel likelihood learns patterns but fails precise conditional writing

The completed 3.453M-parameter writing-head probe does not qualify language
generation from glyphs. Autonomous generation gives **0/128 exact Chinese
development first blocks and 0/64 exact English development first blocks**.
The Chinese training cohort also has no exact blocks; English training has
12/256. These are complete physical first-block measurements, not independent
semantic accuracy or complete-answer scores. All four prospective exactness
criteria fail. Only the positive Chinese question-effect criterion passes.

The [registered protocol](conditional_pixel_density_protocol.md) trains a
normalized causal likelihood over physical 2-by-4 pixel patterns, conditioned
on the existing input-only V68 states. The head sees no answer text, lexical
output inventory, target pixels or retrieved examples during generation.
Its discrete patterns enumerate physical pixels; they do not remove every
discrete variable. The frozen reader is externally pretrained. This test
does not demonstrate a language model learned from images from scratch.

| Cohort | Exact first blocks | Foreground F1 |
| --- | ---: | ---: |
| Chinese training, normal input | 0/256 | 43.485% |
| English training, normal input | 12/256 | 64.047% |
| Chinese development, normal input | 0/128 | 33.498% |
| Chinese development, question ink erased | 0/128 | 29.918% |
| English development, normal input | 0/64 | 37.629% |
| English development, all prompt ink erased | 0/64 | 24.994% |

Chinese normal-minus-erased F1 is **3.580 percentage points**, with a 95%
article-bootstrap interval of **[1.692, 5.671] points**. Both first blocks are
exact in **0/64** Chinese question pairs. Sensitivity to question ink is
measurable, but it does not establish correct language answers. Development
examples are reused; no sealed evaluation set is opened.

The model learns its supplied-prefix image objective. Mean training NLL
falls from **5.7126 to 0.5764 nats per eight-pixel piece**. Final Chinese and
English training NLLs are 0.8207 and 0.3320; development normal NLLs are 2.0057
and 1.4398. These losses condition on correct previous target pieces. The
autonomous scores above use only previously emitted pieces. Likelihood
reduction therefore cannot substitute for accurate free generation.

## Execution, controls and limits

Training completes the fixed 1,200 updates and 75 full epochs over 512
records, retaining 38,400 example draws. A separately counted exact replay
executes one extra update and 32 extra draws. The checkpoint restores before
that replay, so it adds no retained learning exposure. Model parameters,
optimizer state, loss and gradient norm match exactly. Every source target
survives independent lossless packing verification; future target changes
have exactly zero effect on earlier logits. Cached/full-prefix logits differ
by at most 8.941e-7, below the registered 1e-4 tolerance, and their greedy
outputs are bit identical in the registered check.

Active training takes **17.282 seconds**, final generation of all 896 blocks
takes **12.502 seconds**, and the execution including checks and output saving
takes **34.443 seconds**. Peak allocated head VRAM is **0.825 GiB** on GPU0.
These costs exclude the earlier 8B reader training, its 896 cached forwards,
external pretraining and CPU preparation/audit. This is a small-head cost,
not an efficiency comparison with a useful conventional language model. No
new backbone is loaded, GPU1 is preserved, and the model process exits.

Two earlier executions remain failures. The first stops before optimization
on an unsupported deterministic fused NLL kernel. The second completes two
reported updates and reaches the update-three replay, then fails an optimizer
state device comparison. Their checkpoints, logs and
[first](evidence/conditional_pixel_density_v1_failure.json) and
[second](evidence/conditional_pixel_density_v2_failure.json) receipts are
preserved. The completed third execution uses an equivalent explicit
log-softmax likelihood and correct checkpoint placement. No quality threshold,
numerical tolerance, cohort or training schedule is relaxed.

The independent CPU audit recomputes all 896 physical scores, verifies every
PNG against emitted bits, and checks source bindings and training-only mean
statistics. It does not independently replay all neural generations or the
inherited reader. The head consumes detached contextual states, not PNGs.
Those states come from the custom V68 image interface; the source VLM's native
13/16 QA result belongs to a different interface and is not this probe's
performance. Complete streams, learned stopping, novel fonts, historical
forms and general language understanding are not evaluated here.

## Complete original image evidence

All 896 unmodified output PNGs, six complete plates, per-case scores, training
history and provenance are retained in the
[evidence archive](evidence/conditional_pixel_density_result/result.json).
Each plate is displayed at one physical image pixel per file pixel. Rows
within each case show reference, normal generation, and erased-input
generation when present. The complete six plates were visually inspected;
missing or displaced strokes remain common. That unblinded inspection is
separate from the quantitative scores and is not a semantic benchmark.

- Chinese training: [000–127](evidence/conditional_pixel_density_result/plates/train_0000.png), [128–255](evidence/conditional_pixel_density_result/plates/train_0128.png).
- English training: [256–383](evidence/conditional_pixel_density_result/plates/train_0256.png), [384–511](evidence/conditional_pixel_density_result/plates/train_0384.png).
- All development controls: [Chinese](evidence/conditional_pixel_density_result/plates/development_zh_0000.png), [English](evidence/conditional_pixel_density_result/plates/development_en_0000.png).

The result rejects this fixed recipe as a qualified writing component. It
does not establish an impossibility theorem for image-based language
learning or uniquely identify the remaining repair. A sharper image
likelihood alone has not solved conditional language writing. The general
ILM objective remains unachieved, and this result has not yet been promoted
into the manuscript or used to replace its existing figures.
