# Visual Semantic Plan V36 Research Note

Date: 2026-08-14

Status: design evidence collected before V36 implementation or training

## Decision question

V35 proved that a complete raster-input/raster-output causal model can be
trained on one RTX 4090, but it did not learn a reliable relation between a
written prompt and the meaning of its answer. Its local next-patch objective
produced nonblank, prompt-responsive images while autonomous instruction
accuracy remained below the shuffled-prompt control. Scaling that exact loss
would spend more compute on a mechanism that has not learned answer semantics.

V36 therefore asks a smaller question before reopening image generation:

> Can a model read a Chinese prompt only as pixels and predict a continuous,
> answer-level visual semantic plan that identifies the correct held-out answer
> image against controlled alternatives?

This is a necessary condition for an image-native language generator. It is
not sufficient by itself: a planner that retrieves a held-out answer embedding
has not yet written that answer. The raster writer is attached only if this
semantic gate passes.

## Evidence from earlier local models

### V32 did not isolate semantics

V32 combined a PIXEL-M4 reader, causal planner, 32-dimensional glyph state,
and raster decoder. A 4,500-update development run took 195 seconds and 1.21
GiB peak allocated VRAM, but autonomous answer accuracy was 0.053 percent and
the writer overflowed on every evaluated answer. The target raster interface
was too weak to tell whether the planner understood the prompt.

### V34 solved reconstruction, not meaning

V34 qualified a 7.42M-parameter continuous, codebook-free glyph codec. It is a
useful actuator because it preserves modern and historical glyph images. Its
latents are optimized for local reconstruction, however, and are not an
answer-level semantic target.

### V35 learned appearance without semantic binding

V35 trained 129.09M parameters for 22,000 BF16 updates in 2.77 hours on one
RTX 4090. It produced visible closed-loop raster output, but development copy
accuracy was 0.3125 percent and instruction accuracy was 0.1116 percent, below
the shuffled control. The result rejects another local next-patch run as the
next experiment.

## External work that changes the design

External work is used when it provides the strongest measured foundation. Its
origin, license, and exact role remain explicit.

### Pixel sentence representations

[Pixel Sentence Representation Learning](https://arxiv.org/abs/2402.08183)
trains a sentence and document encoder using rendered text, visual
perturbations, topical alignment, natural-language inference, and multilingual
transfer. Its inference representation is a normalized mean over active visual
patch states. This directly addresses the missing answer-level target without
introducing token IDs into the student.

The official
[Pixel-Linguist repository](https://github.com/gowitheflow-1998/Pixel-Linguist)
and
[Pixel-Linguist-v0 checkpoint](https://huggingface.co/Pixel-Linguist/Pixel-Linguist-v0)
do not state a software or weight license as of the date above. V36 may use the
pinned checkpoint for local research, but must not commit, redistribute, or
silently relicense it. The released source is not copied into this repository.

### Raw reconstruction is not a semantic objective

[Autoregressive Semantic Visual Reconstruction](https://arxiv.org/abs/2506.09040)
reports that reconstructing raw appearance can fail to improve, and can impair,
visual understanding, while reconstructing semantic visual representations
improves it. This agrees with the V32 and V35 failure pattern. V36 supervises a
semantic visual state before supervising output pixels.

### Compact pixel readers are plausible

[Text Rendering Strategies for Pixel Language Models](https://arxiv.org/abs/2311.00522)
shows that rendering design changes both efficiency and representation quality,
and that a 22M pixel model can match an 86M baseline on the reported tasks.
V36 does not claim that result transfers automatically, but it supports later
distillation after the mechanism is demonstrated.

### A later writer need not use diffusion

[Fast Autoregressive Models for Continuous Latent Generation](https://arxiv.org/abs/2504.18391)
replaces an iterative diffusion head with a lightweight shortcut head and
reports 2.3 times faster generation than MAR. If V36 passes, the writer
comparison should include a deterministic or shortcut continuous head rather
than assuming diffusion is necessary.

## Pre-protocol foundation audit

The audit used the 197 fixed Chinese Alpaca development records, rendered as
16 by 1,024 grayscale strips. It mapped released `vit.*` tensors into the
current Transformers `ViTModel`, discarded the training MLP for pure active
patch mean pooling, and normalized the result. This compatibility path uses
white padded patches and masks pooling; it does not import the unlicensed
upstream source.

### PIXEL-M4 baseline

The Apache-2.0 PIXEL-M4 reader did not provide an adequate semantic target.
Its CLS vectors were almost identical for all examples. Active-patch mean
pooling was only weakly informative.

| Metric | PIXEL-M4 active mean |
|---|---:|
| prompt-to-answer top-1 retrieval | 1.02% |
| top-5 retrieval | 4.06% |
| mean reciprocal rank | 0.0368 |
| correct minus cyclic cosine margin | 0.00439 |
| correct beats cyclic pair | 50.76% |

### Pixel-Linguist-v0 baseline

One development prompt overflowed the fixed strip, leaving 196 auditable
pairs. Without project training, Pixel-Linguist-v0 produced a measurable
Chinese prompt-answer relation:

| Metric | Pixel-Linguist pure mean |
|---|---:|
| random top-1 chance | 0.51% |
| prompt-to-answer top-1 retrieval | 4.59% |
| top-5 retrieval | 12.76% |
| mean reciprocal rank | 0.0992 |
| correct minus cyclic cosine margin | 0.02413 |
| correct beats cyclic pair | 58.67% |
| same-answer cross-font cosine | 0.86785 |

The checkpoint is not already a question-answering model, but its signal is
more than nine times top-1 chance and materially stronger than PIXEL-M4. It is
therefore a defensible visual semantic foundation and target for V36.

## Selected mechanism

V36 separates reading, planning, and writing.

1. A pinned visual sentence reader converts only the prompt raster and patch
   mask into contextual visual states.
2. Learned plan queries cross-attend to those states and predict one global
   answer plan plus four spatial answer-chunk plans.
3. A frozen copy of the visual sentence reader observes only the answer raster
   during training and supplies detached semantic targets.
4. The deployable planner contains the prompt reader and plan head only. It
   does not contain the answer teacher, candidate answers, OCR, or strings.
5. A raster writer is considered only after the plan identifies held-out
   answers under prompt, font, blank, shuffle, and counterfactual controls.

For prompt raster `x`, prompt mask `m`, frozen answer teacher `T`, student reader
`R`, cross-attention planner `P`, and normalized projection `W`:

\[
H = R(x,m), \qquad U=P(Q,H,m), \qquad z_k=
\frac{W_k U_k}{\lVert W_k U_k\rVert_2}.
\]

The answer target is derived from pixels, not text labels:

\[
q_0 = T(y), \qquad q_k=T(C_k(y)), \quad k=1,\ldots,4,
\]

where `C_k` copies one fixed spatial quarter of the answer strip into a blank
teacher canvas. This preserves a global semantic state and ordered local visual
states without character segmentation or a visual codebook.

For batch size `B`, the global symmetric contrastive term is

\[
\mathcal L_{NCE} = -\frac{1}{2B}\sum_i \left[
\log\frac{e^{z_i^Tq_i/\tau}}{\sum_j e^{z_i^Tq_j/\tau}}+
\log\frac{e^{q_i^Tz_i/\tau}}{\sum_j e^{q_i^Tz_j/\tau}}
\right].
\]

It is combined with cosine regression, a hardest-valid-negative margin,
spatial chunk alignment, two-view prompt consistency, and visual-length Huber
loss. No next-character, next-token, Unicode, OCR, codebook, or raw next-pixel
target enters this stage.

## Why retrieval is used only as an evaluator

The development evaluator compares predicted plans with a fixed bank of
answer-image embeddings because that gives an objective semantic test. The
candidate bank is not an input to `forward` or `generate_plan`, is not included
in the checkpoint, and cannot be used at runtime. Passing retrieval means that
the continuous state selects the right meaning among held-out alternatives. It
does not permit the claim that the model generated an answer.

## Interpretation of outcomes

If the semantic-plan gate fails, the correct action is to improve the visual
reader, alignment data, or planning objective. Another raster writer run is
not justified.

If it passes, V36-R may condition the qualified V34 actuator or a faster
continuous shortcut writer on the frozen plan. The writer must then pass a
second autonomous raster-generation protocol. The semantic planner result and
the writing result remain separate claims.

## Claims deliberately excluded

V36-P cannot establish:

- parity with Qwen, GPT, Llama, or another general assistant;
- autonomous image generation;
- factual reliability outside the fixed data domain;
- a license to redistribute Pixel-Linguist weights;
- superiority of pixels over tokens in total training compute; or
- human-equivalent reading.

It can establish the narrower missing mechanism: a compact image-only runtime
can map a written prompt to a controlled, answer-level continuous semantic
state on one RTX 4090.
