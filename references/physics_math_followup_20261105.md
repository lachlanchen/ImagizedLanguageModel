# Physics and mathematics after the V63 language test

Evidence checked on 5 November 2026. The objective is useful independent
image-input/image-output language on one RTX 4090. No reviewed result proves
a singularity, unlimited self-improvement or general frontier-model parity.
The [execution and mathematical decision update](physics_math_action_update_20261105.md)
supersedes the prospective execution status below and adds an error decomposition
and a transport diagnostic; it does not change the frozen training protocol.
The earlier [review](physics_math_routes_20261105.md) covers transport,
physical-field hierarchies and thermodynamic initialization. This update
connects recent primary work to the completed [V63 result](question_assignment_v63_result.md).

## Recent results and their limits

**Spectral renormalization group, 30 October 2026, v1 preprint.** Martin
proposes a connection between layer spectra, retained covariance volume and
learning. The optimizer experiments include a three-layer MNIST MLP; WW-PGD
changes spectral diagnostics while accuracy remains close to, or slightly
below, AdamW. The paper does not prove optimizer convergence or universal
applicability. This supports candidate collapse diagnostics, not using an
exponent of two as a language-quality target.
[Primary paper, Sections 7–9 and Appendix A](https://arxiv.org/html/2611.00422v1).

**Lagrangian flow matching, 31 August, revised 5 September 2026, v2 preprint.**
Milanfar derives straight trajectories from target invariance using
characteristics. Intersecting characteristics and changing denoiser predictions
explain why empirical generation may still require multiple evaluations.
This motivates measuring endpoint drift in a coherent image writer; it does
not prove one-step glyph accuracy or the emergence of language knowledge.
[Primary paper, Sections 2–6](https://arxiv.org/html/2609.00198v2).

**Multimodal Flow, 30 September 2026, v1 preprint.** The authors study
chunk-causal flow at 0.6B–1.6B parameters and report 150B pretraining tokens.
Semantic visual embeddings outperform raw pixels in their evaluated setting.
The language path uses T5-small token-derived embeddings and a text decoder.
Causal ordering and continuous generation are relevant, but the published
model does not satisfy ILM's image-only deployment boundary or demonstrate
our GPU budget.
[Primary paper, Sections 2.2–2.3 and 3.5](https://arxiv.org/html/2609.40362v1),
[author code](https://github.com/hustvl/Multimodal-Flow).

**Optical Character Grounded Visual Reasoning (OCG), page dated 14 August
2026.** The author-attributed manuscript hosted by alphaXiv reports rendered
questions and generated answer images with separate reasoning and rendering
teachers: 94.6% CLEVR accuracy versus 50.1% without dual-teacher distillation.
Section 4.4 specifies 50,000 steps on four A100 80-GB GPUs with inherited
understanding/generation models. This is particularly close evidence for
transferring both semantics and rendering. Restricted CLEVR answers do not
establish open-ended language or historical writing. No peer-review status or
canonical arXiv identifier was independently verified; these statements come
from the manuscript, not the hosting site's AI overview.
[Primary PDF, Sections 3.3–5.2, Tables 2–3](https://www.alphaxiv.org/abs/2608.ocg-optical-character-grounded-reasoning.pdf).

The 2 November peer-reviewed
[thermodynamic-initialization study](https://www.nature.com/articles/s44335-026-00103-2)
remains relevant to suppressing slow modes in its quadratic Langevin systems.
The earlier review distinguishes its March preprint from November publication.
Neither analog relaxation time nor quadratic-system acceleration is a measured
GPU language-training advantage.

## What the experiment rules out

V63 has 2/256 exact answers for control and 1/256 for aligned; both have zero
fully correct pairs. Aligned has lower strict-raster F1 in the registered
bootstrap. Both nevertheless show resolved normal-minus-blank-question gains.
Detecting any question use is therefore insufficient. Correct conditional
selection and faithful writing must improve together.

The supplied-answer ranking improves descriptively while generation worsens.
A lower energy for an accepted answer is not the ability to write it. This
does not identify whether semantic representation, output calibration, local
image modes, training exposure or their interaction is the dominant cause.

## Mathematical constraints on a useful next actor

For observed prompt image X, answer image Y and deterministic visual state
Z=E(X), a standard identity for finite rasters is

```text
H(Y | Z) - H(Y | X) = I(Y ; X | Z) >= 0.
```

Compression that drops answer-relevant strokes, negations or numbers creates
irreducible uncertainty for a decoder using only Z. The identity neither
certifies an encoder nor supplies a computable training loss. Retaining spatial
detail alongside contextual features is a hypothesis to test, not a theorem
that any chosen architecture will work.

The physical-block factorization

```text
p(Y | X) = product_b p(Y_b | X, Y_<b)
```

requires no character inventory. A flow writer can model a whole block jointly,
but the context model must still choose the correct conditional distribution.
Changing only the sampler cannot supply missing task knowledge.

Causality must also hold before encoding. A bidirectional encoder can see the
whole observed prompt, but must not encode future answer blocks into earlier
history features. Our [executed counterexample](../docs/causal-visual-prefix.md)
shows this leakage. Completed answer blocks need independent or otherwise
causally restricted encoding. Offline teacher targets may use supervision;
the actor input must remain a function of observed pixels.

## Next hypothesis: transfer semantics and learn coherent writing

The strongest candidate is to adapt an already grounded visual-language body
and train a continuous image writer, transferring semantic and rendering
signals separately. This is an engineering inference from the papers and our
failures, not a result established by either.

```text
prompt pixels + completed writing blocks
 -> spatial visual features -> contextual body without vocabulary matrices
 -> conditional continuous image writer + learned stop decision
 -> next physical writing block
```

A possible loss for a new protocol is

```text
U ~ Normal(0,I), t ~ Uniform(0,1)
Y_t = (1-t) U + t Y
L_flow = E ||v_theta(Y_t,t,h) - (Y-U)||^2
L_state = E ||P(h_student) - stop_gradient(h_teacher)||^2
L = L_flow + lambda_state L_state + lambda_stop L_stop
```

Student states depend only on visible image history. Teacher states and targets
can be prepared from text offline with provenance; answer information belongs
in supervision and never in actor conditioning. A new protocol must fix the
alignment, weights, sampling steps, stop rule and trainable layers before its
quality audit. The formula alone establishes neither learning nor efficiency.

The authenticated cached Qwen3-VL weights provide an executed image encoder
and a shape-checked contextual body without vocabulary matrices. Calculated
retained BF16 weights occupy 14.012 GiB. Full-body execution, memory with
activations/writer, adaptation and autonomous quality remain **unverified**.
Frozen inherited weights plus modest adapters are a candidate; a full FP32
optimizer/EMA copy cannot be assumed to fit. External pretraining must remain
visible in any claim about training on one 4090.

Advance only after an actual full-model forward/backward, causal and offline
replay checks, coherent held-out writing, then correct ordinary Chinese/English
answers with matched semantic-transfer and question controls. Reconstruction,
copying and supplied-answer ranking remain separate measurements. Keep reused
development data explicit and sealed final data unopened. Measure quality,
VRAM and end-to-end cost against a named baseline before claiming efficiency.
Spectral or thermodynamic acceleration belongs in a later ablation once the
actor works. The general image-language goal remains open.
