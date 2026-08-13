# Visual Semantic Distillation V37 Research Note

Date: 2026-08-14

Status: design evidence collected before V37 implementation, smoke training,
production training, or V37 development evaluation

## Decision question

V36 showed that a compact image-only planner can be trained and audited on one
RTX 4090, but its answer plans were not useful semantic states. EMA development
top-1 was 1.02 percent, raw top-1 was 2.04 percent, and the conjunctive gate
failed. The post-result audit found two independent causes:

1. occupancy was measured after contrast and noise, so shifted white background
   became active and inflated train answer length from the clean development
   mean of 11.53 patches to 37.31; and
2. frozen visual targets had centered effective rank only 4.77/768, while a
   linear map from clean frozen visual features to a strong semantic space
   reached only 2.04 percent top-1.

V37 asks the smallest successor question:

> Can one shared visual reader, adapted end to end from rendered Chinese, map
> unseen prompt images to a continuous semantic answer plan with useful
> held-out retrieval, cross-font, paraphrase, blank, shuffle, counterfactual,
> and rank evidence, while its deployed path remains image-only?

This remains a representation proof. It is not generated writing. A raster
writer stays closed unless the complete semantic gate passes.

## External work is allowed, with a hard runtime boundary

The project does not need to reject useful public work. It needs to distinguish
three roles precisely:

- an **initialization** may supply visual reading ability;
- an **offline teacher** may construct continuous training targets; and
- the **deployed ILM** must run without either teacher service, strings,
  tokenizer, OCR, answer candidates, or external model calls.

This is standard distillation, not a claim that V37 learned semantics from no
prior knowledge. Runtime independence and provenance are separate properties.

## Research evidence that changes V37

### Rendered text needs an explicitly semantic visual objective

[Pixel Sentence Representation Learning](https://arxiv.org/abs/2402.08183)
trains a pixel sentence encoder with semantic similarity, inference, and
multilingual transfer. The released Pixel-Linguist-v0 checkpoint is directly
compatible with the 16 by 1,024 strip used here. Its upstream README states
that the extra training MLP is intentionally dropped for representation
inference, so V36's use of the 198 `vit.*` tensors and masked mean pooling was
correct. The V36 failure is not repaired by loading the four discarded
`pooler.*` tensors.

The checkpoint and repository state no license. V37 may use the exact pinned
weights only for local research and may not redistribute a derived checkpoint.

### Rendering is part of the learning problem

[Text Rendering Strategies for Pixel Language Models](https://arxiv.org/abs/2311.00522)
shows that visual packing changes semantic performance and that compact visual
models can be competitive when redundant patches are reduced. The 2026
[Design Fundamentals of Pixel Text Representation Learning](https://openreview.net/forum?id=Mps3zbNok4)
reports four relevant findings: variable resolution/font size improves transfer,
natural image grounding prevents visual shortcuts, layout-aware rendering
matters, and multilingual semantic mid-training is necessary. V37 applies the
bounded parts that fit the current proof: varied train fonts and sizes,
layout-preserving contiguous strips, independent views, and direct semantic
mid-training. Natural-image grounding and page-scale variable resolution are
deferred rather than falsely claimed.

[PIXEL-M4](https://arxiv.org/abs/2505.21265) further shows that multilingual
pixel pretraining can produce cross-script semantic structure. It remains a
scientific reference, not the selected V37 initialization: the local fixed
Chinese probe measured only 1.02 percent prompt-to-answer top-1 for its
active-patch mean, below Pixel-Linguist-v0.

### A strong continuous multilingual teacher exists locally

[BGE-M3](https://arxiv.org/abs/2402.03216) maps more than 100 languages and
multiple text granularities into a 1,024-dimensional retrieval space. The
[official model card](https://huggingface.co/BAAI/bge-m3) states an MIT license.
V37 uses the exact local Ollama artifact only to construct detached targets.
The tokenizer, model, service, source strings, and target mean are absent from
the deployed model.

On the exact nonsealed V37/V36 prompt formatting, the fixed 196-pair
development diagnostic gives:

| BGE-M3 target-space diagnostic | Raw | Train-joint centered |
|---|---:|---:|
| prompt-to-answer top-1 | 83.16% | 82.65% |
| top-5 | 93.37% | 92.86% |
| mean reciprocal rank | 0.8772 | 0.8723 |
| correct cosine | 0.7084 | 0.5112 |
| correct-minus-cyclic margin | 0.3271 | 0.5331 |
| development answer effective rank | 77.27 | 76.53 |

The joint mean is computed from all train prompt and answer vectors. Centering
retains almost all retrieval accuracy while removing the corpus-wide direction
and increasing pair separation. The train answer space has centered effective
rank 120.74/1024, far above V36's 4.77/768. V37 therefore distills centered,
L2-normalized fields.

The exact local teacher evidence is:

- model tag: `bge-m3:latest`;
- Ollama manifest SHA-256:
  `7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab`;
- model layer SHA-256:
  `daec91ffb5dd0c27411bd71f29932917c49cf529a641d0168496c3a501e3062c`;
- model layer bytes: `1,157,671,200`; and
- audited local server version: Ollama `0.32.6`.

The `latest` tag is mutable; V37 accepts it only when both content hashes match.

### Collapse must be measured, not inferred from loss

[VICReg](https://arxiv.org/abs/2105.04906) separates invariance, variance, and
covariance terms so noncontrastive representations cannot satisfy the objective
by collapsing. V37 already has candidate contrastive supervision, but V36
demonstrated that low-rank targets and plans can coexist with finite losses.
V37 therefore adds mild variance/covariance regularization and makes centered
effective rank an acceptance condition.

## Selected mechanism

For a rendered image `x`, clean patch occupancy `m`, shared visual reader `R`,
masked mean `M`, and nonlinear semantic head `E`:

\[
h=M(R(x),m), \qquad r=\operatorname{norm}(E(h)).
\]

The same `R` and `E` encode prompt views and answer views. There is no separate
OCR or answer-image encoder. A small residual plan head predicts the intended
answer state from the visual prompt state:

\[
d=\operatorname{norm}(G(r)),\qquad
z=\operatorname{norm}\!\left(r+\alpha d\right),\qquad 0<\alpha<0.5.
\]

The direct residual is intentional. The teacher prompt and answer already share
a strong semantic space; a large unconstrained planner would relearn that
geometry from only 5,822 train pairs. Initializing `alpha` near 0.05 lets the
model preserve visual reading while learning only the answer-directed
correction supported by data.

For raw BGE prompt and answer vectors `b_p`, `b_a` and the train-joint mean
`mu`, detached targets are

\[
u=\operatorname{norm}(b_p-\mu),\qquad
v=\operatorname{norm}(b_a-\mu).
\]

Training uses two independently rendered prompt views and two answer views.
The visual student learns `r_prompt -> u`, `r_answer -> v`, and
`z_prompt -> v`. Candidate banks provide training negatives and evaluator
scores only; they are never an inference input or checkpoint tensor.

## Why this is still an image-native model

During data preparation, source text is rendered and BGE-M3 builds continuous
targets. Inside a student batch, the trainable reader receives only pixel arrays
and geometry masks. At deployment:

```text
prompt pixels + clean visual mask -> shared visual reader
                                  -> semantic state + answer plan + length
```

There is no string, character ID, Unicode ID, token embedding, vocabulary
logit, OCR transcript, retrieval call, or BGE service. The student is
image-native even though its knowledge is distilled from external work, just as
a compact image model remains an image model when trained from semantic labels.

## Efficient one-4090 training

Retaining a full effective-batch computation graph is unnecessary. Each
physical microbatch scores its visual states against 512 deterministic detached
teacher candidates, including every positive, and gradients accumulate to the
fixed effective batch. This supplies broad semantic negatives without keeping
hundreds of ViT graphs resident. Four visual views are concatenated into one
reader forward pass.

The planned deployable model remains below 100M parameters. The text teacher
uses GPU only while constructing the target bank, is unloaded, and is never
resident during student training. This is a concrete efficiency claim about the
experiment, not evidence that ILM is already more efficient than an 8B LLM at
equal capability.

## Interpretation of outcomes

If V37 fails, its measurements distinguish at least four cases:

- low prompt-to-prompt target accuracy: visual reading/semantic distillation
  failed;
- good reading but weak prompt-to-answer plan: the residual relation failed;
- good canonical metrics but weak font/paraphrase controls: the reader learned
  rendering shortcuts; or
- high cosine but low effective rank/control drops: the representation
  collapsed or ignored the prompt.

If every gate passes, V37 establishes a bounded visual semantic substrate and
permits a separately preregistered raster writer. It still does not establish
open-ended generation, factuality, Qwen parity, human-like reading, or compute
superiority.

## Claims deliberately excluded

V37 cannot by itself claim:

- generated answer language or readable output pixels;
- a general Chinese assistant;
- understanding of oracle, bronze, seal, handwritten, or unencoded forms;
- parity with a local 4B/8B text model;
- permission to redistribute Pixel-Linguist-derived weights; or
- that offline text-teacher knowledge originated from visual experience.

Its valid target is narrower and necessary: demonstrate that a compact,
independent image-only runtime can read unseen Chinese prompt images into a
semantically useful, noncollapsed answer-conditioning field on one RTX 4090.
