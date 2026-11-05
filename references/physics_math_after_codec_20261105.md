# Physics and mathematics for learning to read and write in images

Primary-source review checked 5 November 2026. The engineering target is an
independent image-input/image-output language model whose adaptation and
inference fit one RTX 4090. The most useful recent findings concern feature
learning, conditional generation and decoder stability. They supply hypotheses
and diagnostics, not a demonstrated route to an AI singularity. “Singular” in
singular learning theory refers to parameter-space geometry and identifiability;
it does not mean autonomous intelligence escalation.

This note extends the [earlier physics review](physics_math_routes_20261105.md)
and [learning-response analysis](learning_response_and_visual_sufficiency_20261105.md)
using the completed V66/V67 evidence. It does not change completed protocols,
declare a language pass, or replace measured answers with a spectral score.

## Selected recent findings and their limits

| Primary work | What it establishes | Transfer worth testing |
| --- | --- | --- |
| [Hitchcock et al., *Susceptibilities for Neural Networks Learning from Physical Data*, 2 November 2026, v1; author record reports NeurIPS acceptance](https://arxiv.org/html/2611.02632v1) | Bayesian response measurements identify useful training distributions and track collision-resolution capability in two synthetic physical systems. Posterior sampling requires tuned settings. | Test which data develops question-specific behavior. Matched short training interventions are a cheaper engineering alternative, but must not be called the paper's posterior susceptibility. |
| [Bhandarkar and Hanin, *Investigating the Early Signatures of Feature Learning in Deep Linear Networks*, 27 October 2026, v1 preprint](https://arxiv.org/html/2610.37834v1) | In its high-dimensional, deep-linear, one-gradient-step setting, feature learning can improve ridge prediction when task-aligned structure and signal are sufficient. Its advantage vanishes in the isotropic case. | Measure alignment with answer-relevant structure, rather than treating a low-dimensional representation or visible feature change as evidence of generalization. Its asymptotic conditions do not describe our transformer. |
| [Dandi et al., *Neural LoFi with Backward Coupling*, 23 October 2026, v1 preprint](https://arxiv.org/html/2610.32849v1) | Under controlled initialization and surroundings, coupled incoming/outgoing weights amplify singular modes. A spectral algorithm recovers features missed by forward layer-wise learning and improves binary CIFAR-10 experiments. Repeated-correction guarantees remain open. | Retain a gradient path from writing errors into contextual adapters. This motivates coupling; it neither proves rank-eight adapters sufficient nor supplies an established optimizer for our inherited actor. |
| [Pham et al., *Emergence of diffusion models from associative memory through memorization and generalization*, Nature Communications, accepted article published 25 October 2026](https://www.nature.com/articles/s41467-026-78727-y); [full author preprint](https://arxiv.org/html/2505.21777v2) | Relates diffusion generation to memory attractors, with an intermediate regime of spurious states between memorization and generalization. The original preprint was posted 27 May 2025; the journal publication is new. | Separate image novelty from correct conditional answers. New attractors or images unlike training examples need not be valid writing, factual language or a response to the question. |
| [Yang et al., *Large Language Continuous Diffusion Models* (Sigma), 2 October 2026, v1 preprint](https://arxiv.org/html/2610.02665v1) | Demonstrates large continuous diffusion language models, while retaining a tokenizer and categorical readout. Appendix B identifies fragile continuous decoding. Reported 3B/8B training uses roughly 9,600/19,200 GB200 GPU-hours. | Audit the image decoder away from exact encoded targets. Continuous states alone establish neither vocabulary-free inference nor affordable foundation training. |
| [Li et al., *Muon Learns Facts Better*, 2 October 2026, v1 preprint](https://arxiv.org/html/2610.02798v1) | Spectral gradient flow balances subject/relation learning in a linear factual-recall model. A 35M Pythia experiment supports faster subject learning, with similar eventual accuracies across optimizers. | A later, matched optimizer ablation may test whether rarely learned relations improve sooner. Orthogonalizing low-rank adapter factors is not equivalent to the theorem's full-matrix update. |
| [Pracher et al., *A Spectral Theory of Grokking: Weight Decay induces Feature Learning*, 22 September 2026, v1 preprint](https://arxiv.org/html/2609.26679v1) | Studies residual-driven feature growth under homogeneous networks, squared loss, coupled decay and gradient flow; modular-addition transformer experiments support some macroscopic predictions. | Track held-out capability alongside fitting. The theory does not justify waiting indefinitely for ordinary language to “grok,” nor transferring its decay scaling directly to AdamW. |
| [Jeon et al., *Latent Reasoning in Continuous Space for Unified Multimodal Models* (LARC), 30 October 2026, v1; author record reports NeurIPS proceedings](https://arxiv.org/html/2610.42149v1) | Interleaved language/latent reasoning improves image generation on BAGEL. Training retains next-token supervision and special tokens; Appendix A reports eight A100s and approximately 300 plus 96 GPU-hours. | Continuous internal reasoning is relevant, but these results do not qualify vocabulary-free image-language inference or one-4090 foundation training. |
| [Nguyen and Barbier, *Sharp feature-learning transitions and Bayes-optimal neural scaling laws in extensive-width networks*, 11 May 2026, v1 preprint](https://arxiv.org/abs/2605.10395) | A noisy teacher–student model links effective learnable width to data. Its Bayes-limit analysis uses a heuristic decoupling argument supported numerically; an Adam-trained student approaches the predicted laws. | Budget useful capacity against available data. The result does not specify an adequate adapter rank or guarantee our visual transformer reaches the Bayes limit. |

These are selected relevant results through the review date, not an exhaustive
survey or independent reproduction of their experiments. Publication dates,
preprint dates and demonstrated hardware budgets are kept distinct.

### Additional findings: physical feedback and learning landscapes

The following primary sources were checked on the same review date. Their
usefulness lies in particular design constraints, rather than a demonstrated
breakthrough in one-GPU general intelligence.

| Primary work | Relevant finding and boundary | Decision for ILM |
| --- | --- | --- |
| [Hong, *Predictive Latents for Autoregressive Language Models: Possibility, Gauge, and Obstruction*, 20 October 2026, v1 preprint](https://arxiv.org/html/2610.28540v1), Sections 3, 6–7 | Distinguishes prediction on supplied histories from iteration on generated states. Constant states illustrate why a small representation loss can carry no predictive information. Its formal setup concerns finite-alphabet language and a latent transition, not our thresholded pixel-feedback actor. | Measure actual autonomous answers. Normalized states and small one-step errors are insufficient qualification. Use the elementary distinction; do not treat the paper's other theorem statements as a verified certificate for this actor. |
| [Guo et al., *Cola2: Co-Adaptive Hierarchical Language Modeling with Continuous Semantic Compression and Balanced Discrete Generation*, submitted 22 October 2026, v1 preprint](https://arxiv.org/pdf/2610.31305), Sections 3.2–3.3 | Re-encodes emitted text to keep continuous history consistent with what was generated. It combines predictive and reconstruction supervision, retains a discrete text decoder, and reports a 346.03-billion-source-token run with inherited initialization. | Our corresponding image-only operation already re-encodes emitted binary image blocks. Preserve this physical feedback. The external result motivates consistency, but does not demonstrate our writing quality, vocabulary-free language or affordable foundation training. |
| [Ersoy, Cardozo Licha and Wiesner, *Phase transitions reveal hierarchical structure in deep neural networks*, Communications AI & Computing, 21 October 2026](https://www.nature.com/articles/s44488-026-00021-z) | Uses a quadratic regularizer to probe transitions and connections between learned solutions. Experiments include MNIST, Fashion-MNIST and a small synthetic convolutional setting. | Consider a later controlled regularization ablation only if optimization is the measured failure. A transition or flat path is not evidence of useful language. The study does not supply a transformer training-speed guarantee. |

There are reasons to check even a relevant mathematical preprint closely.
For example, Hong's Theorem 17 gives squared rollout risk proportional to the
horizon H, while its stated Theorem 16 bound at unit Lipschitz constant is on
root-mean-square error and grows as H. The former alone does not establish
sharpness of the latter. This is a direct algebraic reading of those statements,
not an independent verification of the paper. No implementation decision here
depends on that sharpness claim.

For a quadratic penalty written as beta times the squared Euclidean norm,
its Hessian contribution is **2 beta I**. A beta I convention instead uses a
one-half factor in the penalty. Regularizer formulas must keep this convention
consistent, and an objective penalty must not be silently identified with
decoupled AdamW decay. Neither this identity nor a landscape visualization
specifies an effective learning rate or decay for our adapters.

The 2 November
[thermodynamic-initialization result](https://www.nature.com/articles/s44335-026-00103-2)
already discussed in the [earlier review](physics_math_routes_20261105.md)
concerns relaxation of physical computing systems. Its optimized preparation
is relevant to that hardware setting. It provides no measured acceleration of
our digital GPU training. A physics-inspired replacement should earn its place
through matched language quality and total local compute, including preparation.

The LARC comparison is especially relevant to continuous image generation,
but its GenEval improvement is not a writing-comprehension result. A later
bounded latent-computation ablation would need correct autonomous writing
first, identical data and weights, and total latency as well as answer quality.
It should not replace the present causal experiment. Its own reward signal is
not independent evidence that the generated answer is correct.

## The local evidence changes the priority

The [V66 direct writer](direct_writer_v66_result.md) completes all 10,000
updates, with 56.563% foreground F1 and 3/128 exact development blocks. Its
conditioning advantage passes, but its original 90% fidelity requirement fails.
It is reconstruction from supplied image features, not language generation.

The [V67 codec audit](codec_output_v67_result.md), on the same 128 reused
development blocks, reaches 99.998% F1 and 126/128 exact blocks. Shuffled codes
give 21.413% F1. Every source image and all 256 outputs were independently
verified. This qualifies a particular output representation at supplied codes;
it does not show that prompt-conditioned states land in that representation.

The evidence favors using separate reading and writing representations. A
reader may discard font details usefully for answering, while a writer needs
precise spatial distinctions. Learning an inverse of the reader is unnecessary
when a separate image codec already preserves the physical writing.

## A simple mathematical model

For prompt pixels X and completed answer blocks Y_<t, define

\[
 h_t = F_\theta(E_r(X),E_r(Y_{<t})),\qquad
 \widehat z_t = N(A_\theta(h_t)),\qquad
 \widehat Y_t = D_w(\widehat z_t).
\]

E_r is the visual reader, F is the causal contextual body, A is a small learned
projection, N is normalization, and D_w is the frozen writing decoder. Training
targets are detached continuous codes z_t = E_w(Y_t). The physical image
block has 32-by-64 pixels; its writing state contains two 768-vectors. The
vectors are computed from images, not indices in a character inventory.
At inference, only the prompt and generated history are available.

The prepared candidate uses the straightforward objective

\[
 L = \operatorname{MSE}(N(\widehat z_t),N(z_t))
   + \operatorname{MSE}(D_w(\widehat z_t),Y_t)
   + L_{\mathrm{stop}}.
\]

This is a regression objective, not a normalized language likelihood or a new
learning theorem. The fixed decoder still transmits gradients:

\[
 \nabla_\theta L_{\mathrm{image}}
 = J_{\theta\to z}^{\mathsf T}
   J_{D_w}(\widehat z_t)^{\mathsf T}
   \nabla_Y L_{\mathrm{image}}.
\]

Freezing decoder weights does not mean detaching its input. The output loss can
therefore train contextual adapters without paying optimizer-state memory for
all decoder or inherited backbone parameters. The implementation normalizes
the projection before decoding and normalizes both states again for the state
loss, as written above. Its complete-model preflight now passes all 24 real GPU
checks, including backward, causality and exact resume, at 16.953 GiB allocated
peak. This verifies execution after the interface change; its two-update
fragmented writing does not qualify language.

For a small latent error delta, differentiability gives a local expansion

\[
 \|D_w(z+\delta)-D_w(z)\|_W^2
 = \delta^{\mathsf T}J_{D_w}(z)^{\mathsf T}WJ_{D_w}(z)\delta
   + o(\|\delta\|^2).
\]

Here W can be the fixed pixel-MSE normalization. This familiar pullback metric
explains why equal latent MSE can produce unequal stroke damage. It is local;
it supplies no global robustness certificate. Thresholding is discontinuous,
so continuous pixel error and exact writing must both be measured. A large
average image score can coexist with lost decisive strokes.

There is a second constraint: for an unrestricted predictor trained on latent
squared error alone, the optimum is E[z|X,Y_<t]. If the conditional writing distribution has
separated modes, its mean need not decode to a valid or intended form. Good
codec reconstruction therefore does not prove that deterministic latent
regression is a sufficient language generator. The decoded-image loss changes
that optimum; the conditional-mean identity does not characterize the complete
three-term objective. Test this simple candidate
first; introduce a conditional multimodal density only if actual generated
answers reveal that failure. Earlier failed flow experiments remain evidence
against treating diffusion itself as the missing explanation.

## Executed diagnostic: perturbing the real writing space

The [protocol](continuous_writing_geometry_protocol.md) was committed before
execution. The [audit](../scripts/audit_continuous_writing_geometry.py) runs
only on the 64 existing V65 training images, without updates or GPU use. Four
seeded, normalized tangent directions per block give 256 cases at each level.
The codes are re-normalized after perturbation; sigma is the pre-normalization
coordinate scale, not the resulting latent RMS error. Original references,
all code tensors and raw decoded images are preserved. The independent
[verifier](../scripts/verify_continuous_writing_geometry.py) replays the native
decoder and recomputes all 1,536 perturbation scores. Its
[verification receipt](evidence/continuous_writing_geometry_result.json)
records all checks passing.

| Perturbation scale sigma | Actual mean latent MSE | Foreground F1 | Exact blocks / 256 |
| --- | ---: | ---: | ---: |
| 0 | 1.23e-9 | 100% | 256 |
| 0.01 | 1.00e-4 | 100% | 256 |
| 0.03 | 8.99e-4 | 100% | 256 |
| 0.1 | 9.93e-3 | 99.9986% | 253 |
| 0.3 | 8.43e-2 | 99.7517% | 73 |
| 1.0 | 5.86e-1 | 88.3693% | 0 |

The original, unperturbed codes reconstruct all 64 source blocks exactly.
Re-normalization at sigma zero has negligible floating-point effects. The
diagnostic takes 4.577 seconds on four CPU threads, excluding Python startup;
it is not a training-throughput or GPU-efficiency comparison. The raw report
is `artifacts/continuous_writing_geometry_20261105/attempt01/report.json`,
SHA-256 `e1d43aa1f5651cf106d81d0927e8afb451f33493f8c88789a4a13abfded408d9`.

To reproduce in the project's compatible PyTorch environment, choose a fresh
attempt directory; completed or interrupted attempts are never overwritten:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python scripts/audit_continuous_writing_geometry.py \
  --out artifacts/continuous_writing_geometry_20261105/new-attempt
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python scripts/verify_continuous_writing_geometry.py \
  --run artifacts/continuous_writing_geometry_20261105/new-attempt \
  --out artifacts/continuous_writing_geometry_20261105/new-attempt/verification.json
```

These sampled local directions support keeping the codec frozen for the next
experiment. They do not cover worst-case or learned prediction errors, unseen
fonts, free-running history, valid language or historical forms. In particular,
the high F1 at sigma 0.3 hides an exact-block rate of only 28.5%.

## From a writing codec to a complete answer

The current actor's feedback path is concrete: it thresholds a decoded block,
reads those emitted pixels through the visual reader and appends their features
to the causal context. It does not simply feed the predicted writing code back
as if it were a reader state. This respects the separate spaces used for
reading and writing. It also means that a smooth latent-transition theorem
does not automatically describe the complete generator: pixel thresholding
and learned stopping introduce discontinuities.

An elementary condition makes the remaining requirement explicit. Fix a
prompt and one reference answer of T binary blocks. Let y_ti in {-1,+1}
be reference pixel i in block t. Let a_ti be the actor's continuous pixel
score when its actual inference path is supplied the **exact reference
history**, with no future blocks. Define

\[
 m_t=\min_i y_{ti}a_{ti}.
\]

If every m_t is strictly positive, the actor's hard threshold reproduces each
block on that history. If its computed stop probability is below 0.5 before
each required block and at least 0.5 at the terminal decision, induction
then gives the complete exact answer under autonomous generation: the first
block is exact, so the second sees the exact history, and so on. This is a
sufficient condition, not a new learning theorem, a necessary condition, or
evidence that our trained actor satisfies it. It assumes the same deterministic
inference arithmetic and cache path for the reference-history comparison.
Using the actual probability comparison matters because finite-precision
sigmoid evaluation can round a small negative logit to probability 0.5.

A supplied-code measurement addresses only part of this condition. With
target writing code z_t and decoder D_w, set

\[
 m_t^{\mathrm{codec}}=\min_i y_{ti}D_w(z_t)_i.
\]

If this margin is positive and the predicted state on the exact history obeys

\[
 \|D_w(\widehat z_t)-D_w(z_t)\|_\infty
 < m_t^{\mathrm{codec}},
\]

then the predicted block has the correct signs. This follows from the triangle
inequality, without estimating a global decoder Lipschitz constant. It is a
diagnostic sufficient condition; the difference must be evaluated at actual
predicted states. Random perturbation directions from the existing codec audit
do not establish this condition. The autonomous audit remains the primary
measurement, and its fixed scores do not change.

For a probabilistic generator, the chain rule similarly gives

\[
 P(E_1,\ldots,E_{T+1}\mid X)
 =\prod_{t=1}^{T+1}P(E_t\mid X,E_{<t}),
\]

where E_t includes the required no-stop decision and exact block for t <= T,
and E_{T+1} is the terminal stop. This requires no independence assumption.
For illustration, if each conditional success probability were 0.99, forty
such decisions would jointly succeed with probability about 0.669. Our
supplied-code block scores are not these conditional probabilities. A high
average reconstruction F1 therefore cannot be substituted for whole-answer
success or used to forecast it.

These observations motivate the [completed diagnostic](codec_language_v68_diagnosis_result.md)
after the frozen autonomous audit failed: compare the same final actor's initial
block, exact-history next blocks, generated-history blocks and stop decisions.
Any additional model diagnostic needs its own fixed scope before execution;
it must preserve the original outputs and cannot revise their scores. Poor
first-block selection cannot be blamed on accumulated generated history.
Conversely, good exact-history writing with poor autonomous writing would
motivate a controlled generated-history training intervention. Neither outcome
was established when that diagnostic was registered. Its 32-case measurements
now show poor exact-history prediction even on training examples, and no exact
first block, despite faithful supplied-code reconstruction. Rollout drift is
therefore insufficient as the sole explanation. This does not prove that a
conditional density, a larger readout or a particular optimizer is the repair.

## The next discriminating experiment

1. Preserve the [completed V68 training](codec_language_v68_training_result.md),
   its final checkpoint and all 20,000 journaled draws. The registered budget,
   codec, objective and sources remain unchanged.
2. Preserve the [completed autonomous audit](codec_language_v68_result.md):
   all 384 raw streams are independently verified. Chinese exactness is 0/128
   and F1 is 32.6713%, despite a resolved 13.1628-point question effect. The
   language gates fail. These are reused development data; the genuinely
   sealed final split remains unopened.
3. Use the completed exact-history diagnosis to constrain the next change.
   Poor initial and training-set predictions rule out generated-history drift
   as a sufficient explanation. Measure whether improved conditional features,
   readout capacity or a joint output distribution repairs writing, with
   matched data and original autonomous quality requirements. The diagnostic
   does not select a unique successful algorithm.
4. Treat spectral optimization as a subsequent equal-budget ablation, with
   effective adapter updates and inherited pretraining costs disclosed. Do not
   combine a new optimizer, data mixture and decoder in one comparison.

The new work strengthens the rationale for this bounded experiment. The
required result remains useful unseen prompt-to-answer images from an
independent actor. It does not become a singularity claim, a frontier-parity
claim or a claim that foundation training has been reduced to one GPU.
