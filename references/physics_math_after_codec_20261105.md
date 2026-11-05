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

## The next discriminating experiment

1. Preserve the completed actor preflight and the same live V68 trainer. The
   registered instruction stage uses all 12,976 admitted examples and a fixed
   10,000-update budget. The codec, sources and objective remain unchanged.
2. Execute the already frozen autonomous audit after training finishes. It
   generates 192 prompt-only answers and 192 matched question/prompt-blank
   controls, preserves full raw streams and stopping, and independently
   verifies scores. Chinese cases include two distinct questions per passage.
   CPU evaluator checks pass; no language output has yet been scored. The
   128 reconstruction cases and this instruction audit are development data;
   genuinely sealed final data remains unopened.
3. Diagnose failure before changing architecture. Compare predicted-code
   errors with physical stroke errors. If answers ignore the question, test
   a matched data intervention. If predicted states leave useful writing
   regions, test an output-distribution change. Neither is established yet.
4. Treat spectral optimization as a subsequent equal-budget ablation, with
   effective adapter updates and inherited pretraining costs disclosed. Do not
   combine a new optimizer, data mixture and decoder in one comparison.

The new work strengthens the rationale for this bounded experiment. The
required result remains useful unseen prompt-to-answer images from an
independent actor. It does not become a singularity claim, a frontier-parity
claim or a claim that foundation training has been reduced to one GPU.
