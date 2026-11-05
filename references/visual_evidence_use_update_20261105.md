# From visible writing to usable language

Primary sources checked on 5 November 2026. The immediate problem is learning
question-dependent answers from writing images on one RTX 4090. Recent results
help distinguish visual recognition, accessible language representations and
coherent generation. They do not establish a route to an AI singularity.

## New evidence closest to the bottleneck

| Primary source and version | Reported result | Consequence for ILM |
| --- | --- | --- |
| [Shi, *VLMs Read Aloud*, v2, 3 November 2026](https://arxiv.org/html/2611.01517v2) | In the tested Qwen3-VL-8B setting, recognition saturates near 15-pixel characters, while answering from images catches up only near 24 pixels. On recognized questions at 15 pixels, answering from its own transcript exceeds answering from the image by 10.7 percentage points. The proposed ALOUD system uses language tokens to make recognized evidence usable. | Recognition and evidence use need separate measurements. A pretrained vision encoder alone cannot be assumed to solve conditional language. ALOUD's text intermediate would violate the intended deployed image-only interface; its diagnostic distinction remains useful. The reported resolutions are model/task-specific, not Chinese rendering specifications. |
| [Tang et al., *Beyond Clean Accuracy*, v1, 16 October 2026](https://arxiv.org/html/2610.24290v1) | The tested PIXEL models show delayed, learning-rate-sensitive downstream adaptation and greater dependence on labelled coverage than the text controls. Successful adaptation gives stronger typographical robustness and crosslingual transfer in the studied settings. | Check task accessibility before increasing model size. The evidence concerns downstream classification and transfer, not free answer-image generation. A news summary dated 4 November is not a new version of the paper. |
| [Zhang et al., *PixelBench*, v1, 2 November 2026](https://arxiv.org/html/2611.02540v1) | Existing language models generate palette-index grids through their text vocabulary. Pixel Reconstruction Tuning supplements general visual instruction training and improves the reported Molmo2-4B visual benchmark average. Higher output resolution can reduce quality. | Pixel reconstruction is a plausible auxiliary signal. This interface still generates text-serialized indices and does not demonstrate an independent vocabulary-free language actor. Its reconstruction and instruction mixture is more relevant than adopting numerical JSON as the final writing interface. |
| [Zhao et al., *Numerical Visual Chain-of-Thought*, v2, 15 October 2026](https://arxiv.org/abs/2602.23959v2) | Gaussian or Laplace coordinate policies replace categorical coordinate actions in a multimodal language model. The revised author record reports improved visual grounding and NeurIPS acceptance. | Continuous actions can be trained with familiar supervised and policy objectives. Bounding-box generation is not glyph generation, and the rest of the model retains language machinery. The October publication listing should not be confused with its February first submission. |

These are author-reported experiments, not local replications. Their deployment
boundaries matter: none supplies the complete image-input/image-output actor
required here. In particular, a strong visual transcriber or numerical image
serializer cannot be counted as a useful ILM by changing the task definition.

## Physics and mathematics: what transfers

The [2 November susceptibility study](https://arxiv.org/abs/2611.02632v1)
measures responses to data-distribution perturbations in physical learning
problems. The [July statistical-physics analysis](https://journals.aps.org/prx/abstract/10.1103/56sb-pdh6)
derives feature-specialization transitions for matched teacher/student
multilayer perceptrons. Their relevance is to capability-sensitive data and
optimization diagnostics. Neither derives a training-compute reduction for
our transformer. The [existing theory review](learning_response_and_visual_sufficiency_20261105.md)
states their assumptions and connects them to the current conditional test.

A useful mathematical target is prediction-sufficient compression. With full
observed image X, answer image Y and deterministic state Z = f(X), the optimal
expected log-loss increase from using Z instead of X is

\[
H(Y\mid Z)-H(Y\mid X)=I(Y;X\mid Z).
\]

This identity assumes discrete finite images and an unrestricted optimal
predictor. It says what information must survive; it does not make that
predictor easy to learn. Reconstructing X need not make its useful features
accessible to a small downstream readout. Conversely, discarding a decisive
stroke can lose information that no reasoning head can restore.

For a compact actor, the corresponding model family can be written

\[
p(Y\mid X)=\int p_\psi(Y\mid z)q_\phi(z\mid X)\,dz.
\]

Here z may be a continuous sequence or field, with no character dictionary.
This is a model definition, not a new algorithm or a claim of efficiency.
An image likelihood can train both factors, while an offline teacher can
provide additional training targets. Neither a continuous latent space nor
teacher-feature agreement guarantees the right conditional answer.

## A testable bridge after the current comparison

The current V63 comparison must finish before selecting another intervention.
Its shared-paragraph question swaps and autonomous generation directly test
the distinction raised by the new evidence. No running weights, data,
protocol or quality gates change as a result of this review.

If free writing still fails, the next candidate is **causal language-state
transfer with a retained physical writing path**:

1. Use the already prepared complete Chinese/English images and actual answer
   images. Fix source splits and preserve all ordinary spaces and strokes.
2. Cache a capable teacher's training-only causal states at explicitly aligned
   positions. Each target must document exactly which question and answer
   prefix the teacher saw. A state conditioned on future answer content cannot
   become an actor input at inference.
3. Train visual states and a contextual language body jointly with answer-image
   likelihood and the fixed causal targets. Evaluate whether the transferred
   states distinguish changed questions before adding more rendering capacity.
   Earlier V37–38 pooled semantic targets failed; another pooled embedding
   loss alone would not constitute a new solution.
4. Preserve a direct route for fine visual evidence to the writer. Semantic
   compression can otherwise remove unfamiliar forms that the output needs.
   The writer must generate pixels, with a measured ability to produce forms
   outside its training alphabet.
5. Export only the learned actor. Its inference receives images and previous
   generated images; it must not invoke a tokenizer, OCR, text decoder or
   external teacher. Compare autonomous answers, question interventions and
   complete-line rollouts at a fixed measured GPU budget.

This is a candidate experiment, not an implemented or adopted architecture.
The cached Qwen weight-source inspection establishes parameter compatibility
only; it does not establish visual alignment, full-forward equivalence or a
training fit. A new precision/optimizer design and real backward-memory
preflight are required before using those trunks. Offline teacher costs and
external pretraining must remain separate from local adaptation costs.

The research priority is an accessible, question-sensitive visual language
state coupled to reliable writing. Useful independent answers are the decisive
result; lower reconstruction loss, a physics analogy and a continuous internal
representation are supporting components.
