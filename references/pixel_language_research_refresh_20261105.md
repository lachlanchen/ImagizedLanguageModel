# Pixel-language research refresh

Checked 2026-11-05 while the preregistered V55 arms were training. This note
does not change their model, data, budget or evaluation. It updates the
external evidence considered for subsequent work.

## Directly relevant external results

**MIXAR v2, 15 October 2026.** The revised paper replaces its earlier
adversarial stage with rollout-based GRPO and evaluates free generation. Its
477M stage-2 model reports 34.6% English bAbI and 1.4% Chinese bAbI; the latter
falls from 1.8% at stage 1. This is relevant pixel-language prior work and a
potential pretrained baseline, not evidence that Chinese generation is solved.
The original v1 descriptions in earlier frozen protocols remain historical.
[Author paper](https://arxiv.org/html/2604.11575v2).

The author release supplies a 477M, 32-pixel-patch checkpoint with bundled
rendering configuration and fonts. Its card specifies Apache-2.0 and recommends
the stage-2 weights for free generation. The repository URL in the paper
returned 404 from both GitHub's API and the web fetch on this date, but the
Hugging Face release API returned 200. Pin that release and inspect its custom
code before execution; use the same shared weight cache, not a second copy.
[Author checkpoint](https://huggingface.co/TYTTYTTYT/llama-pixar-477m-p32-stage2).

Pinned revision: `c5d5bcfbad2f290296fd83810558f396c6a6333a`.
Weights: 1,908,608,544 bytes, expected SHA-256
`be5706028246c147998a141110fcc29c9ce2026623b31bd85d20ed0916659af8`.
This pretrained baseline includes external pretraining. Its cost cannot be
reported as local single-4090 training. A local fine-tune would be a separate
measured contribution with that initialization disclosed.

**Parallel Rollout Approximation, June 2026.** PRA uses a learned lower-dimensional
intermediate and pixel decoder, and supplies inference-like pixel feedback
during parallel training. Its prefix-aware representation target is relevant
to the gap between local reconstruction and contextual future-image choice.
The reported benchmark is image generation, not written-language reasoning.
Its numerical ImageNet results cannot qualify our language task.
[Author paper](https://arxiv.org/html/2606.27978v1).

**Hot-Start Chinese Language Modeling, May 2026 revision.** Visual glyph input
improves early learning but converges to essentially the same reported final
accuracy as its index-input baseline. The input-side inductive bias therefore
does not establish better final capacity or image-output language generation.
This distinction matters when interpreting our raster reader's ranking gains.
[Author paper](https://arxiv.org/html/2601.09566v4).

## Implications for the local experiment

The independent deployment contract permits disclosed public pretraining.
Evaluate a released multilingual pixel backbone before assuming that a larger
local reader must be pretrained from scratch. A baseline must use its exact
renderer and retain raw images. Font substitution, OCR-assisted answers and
post-generation image replacement would confound that comparison.

The current V54 probability law already models pixel dependence exactly.
V55 first tests joint learning through that law. A future latent or rollout
intervention should address a measured failure that remains after this control.
If a pretrained baseline gives useful reading or generation, reuse that
component and quantify the local adaptation; do not relabel its foundation
training or external algorithm as a new result.

Neither image synthesis nor a continuous interface alone implies knowledge,
syntax, instruction following or greater task-level efficiency. Compare
held-out answers and coherent visible continuations at matched quality, record
the full initialization and local compute, and keep the final student runtime
independent of the teacher.

## Addendum checked 9 November 2026

[MPIXEL, submitted 21 October](https://arxiv.org/html/2610.29681v1), reports
86M/307M pixel encoders, five downstream tasks across 83 languages, and 120B
unique pretraining patches for its sixteen-language models. These use ViT-MAE;
the reported results do not establish free image-answer generation. Appendix
A.3 reports approximately 1,350/2,500 H100 GPU-hours for base/large pretraining.
The authors' linked repository returned 404 through the current web fetch;
release availability is not established here.

This supports investigating smaller pretrained visual readers, while leaving
their writing capability and local adaptation cost untested. It supplies no
evidence that changing the representation alone removes foundation-training
cost. No weights are downloaded, no second model job is started, and no change
is made to the running complete-image diversity protocol. Its final outputs
must determine whether the next problem is conditional selection, glyph
emission or continuation from the model's own images.
