# Learning responses and sufficient visual representations

Primary-source review checked 5 November 2026. The engineering question is
whether an independent image-language actor can learn useful conditional
writing within one RTX 4090's memory and a measured training budget. Recent
theory offers tests and design constraints; it has not established a route
to an AI singularity or frontier-model performance on this hardware.

## Recent results worth using

| Primary source and date | Result and scope | Consequence to test in ILM |
| --- | --- | --- |
| [Hitchcock et al., *Susceptibilities for Neural Networks Learning from Physical Data*, v1, 2 November 2026](https://arxiv.org/html/2611.02632v1) | Local Bayesian response measurements help select data and track capabilities in percolation and granular-material models. Posterior-sampler settings need tuning; natural-language transfer is untested. The author record reports NeurIPS acceptance. | Measure which training-data variations develop question dependence. Ordinary input ablations or a regression probe are cheaper diagnostics, but are not the paper's Bayesian susceptibility estimator. |
| [Barbier et al., *Statistical Physics of Deep Learning*, PRX, 22 July 2026](https://journals.aps.org/prx/abstract/10.1103/56sb-pdh6) | Bayes-optimal analysis of matched teacher/student multilayer perceptrons finds feature-specialization transitions and suboptimal solutions that can attract practical algorithms. Width scales with input dimension and data are comparable to parameter count. The preprint dates to October 2025. | Model capacity and decreasing loss do not ensure the intended feature has been learned. Compare explicit behavioral controls at fixed data and compute. The result supplies no transformer sample-complexity or one-GPU language guarantee. |
| [Shen and Dong, *A Graph Sufficiency Perspective for Neural Networks*, ACM TOPML, 16 September 2026](https://doi.org/10.1145/3816434), [author preprint](https://arxiv.org/abs/2507.10215) | Studies preservation of target-conditional information using dense-anchor asymptotics or finite-width region-separation assumptions. Its preprint began in July 2025. | A compressed image state should retain answer-relevant distinctions. The paper's assumptions have not been verified for our writing distribution; it does not certify an arbitrary visual encoder. |
| [Zhou et al., *Condition Errors Refinement in Autoregressive Image Generation with Diffusion Loss*, 2 February 2026, ICLR 2026](https://arxiv.org/html/2602.07022v1) | Analyzes patchwise denoising and condition refinement, including an optimal-transport method, under its stochastic-process assumptions; experiments concern natural images. | Generated-history drift deserves a separate rollout test. Its convergence analysis cannot be substituted for correct answers, and does not automatically apply to our Bernoulli-mixture writer. |

The [earlier physics/transport review](physics_math_routes_20261105.md)
separately covers recent lattice-field hierarchies, one-step generation and
thermodynamic computing. Their sampling or hardware results must not be
reported as language-training speedups on a digital GPU.

## The representation question can be stated exactly

Let X be the complete observed writing image, Y the desired answer image, and
Z = f(X) the actor's internal representation. For discrete finite images and
an unrestricted optimal probabilistic predictor, the extra expected log loss
caused by replacing X with Z is

\[
  H(Y\mid Z)-H(Y\mid X)=I(Y;X\mid Z)\geq0.
\]

This is an elementary conditional-entropy identity, not a new theorem or a
result measured on ILM. It gives a useful target: remove visual redundancy
while retaining information needed for the requested answer. Pixel
reconstruction alone optimizes a different requirement. A state that
preserves a page's appearance may still be hard to use for answering a
question; a compressed semantic state may instead erase a decisive stroke.

For a source string S rasterized by a deterministic map R, if R is injective
on the relevant source support, then observing R(S) retains the information
in S. If the renderer collapses two distinctions that change the correct
answer, no downstream predictor can recover that lost distinction reliably
from that image alone. Neither statement implies that learning an efficient
predictor from pixels is easy. They explain why a vocabulary-free interface
is possible in principle, while complete rendering and prediction-relevant
features still matter. Direct historical handwriting images do not require
that an S or a Unicode mapping be available.

For the present actor, physical image blocks provide the numerical interface;
they are not entries in a language alphabet. Its contextual parameters must
still learn order, reference, negation and the requested relation between
question and answer. Compressing those blocks is useful only if those
distinctions survive. Larger images and longer internal traces do not by
themselves establish this property.

## Conditional interaction is the immediate bottleneck

V62 improved held-out bitmap likelihood while both matched actors generated
zero exact primary answers. This is local evidence that improving rendering
density at that budget did not solve the requested conditional task. It is
not a proof against image-language learning.

For two question images X0, X1 about one paragraph and distinct answer images
Y0, Y1, let eij be the mean negative log likelihood of Yj after Xi. V63 trains
the interaction

\[
 D=e_{01}+e_{10}-e_{00}-e_{11}.
\]

Pure question-only and answer-only additive energy offsets cancel. This
cancellation is exact algebra; it is neither a semantic-understanding theorem
nor a new susceptibility measurement. If the target images contain the same
number n of scored pixels, the margin also has the likelihood-ratio form

\[
 D={1\over n}\log
 {p(Y_0\mid X_0)p(Y_1\mid X_1)\over
  p(Y_1\mid X_0)p(Y_0\mid X_1)}.
\]

When lengths differ, each target log probability has its own normalization;
the current D is a length-normalized interaction energy, not this single
joint-assignment log odds or a mutual-information estimate. V63 retains its
registered definition unchanged.

An interaction may still use a superficial cue. For example, if r(X) predicts
answer width and a(Y) is the measured width, the squared-error energy
e(X,Y) = (r(X)-a(Y))² gives

\[
 D=2(r(X_0)-r(X_1))(a(Y_0)-a(Y_1)).
\]

This can rank pairs without generating any writing. Consequently,
autonomous answer images and meaningful question interventions remain the
decisive evidence.

## Implemented follow-through

The [training-only nuisance audit](question_assignment_nuisance_audit_result.md)
implements the last check without changing V63. Fixed, article-isolated
five-fold probes see only question width and coarse ink statistics and predict
physical answer width. Across all 3,793 admitted pairs, context-weighted
assignment accuracy is 51.50% for width alone and 53.31% for width plus ink
moments; a constant predictor ties every pair, scoring 50%. These are
descriptive associations, not uncertainty-adjusted effects or model results.

V63 is the current experiment addressing conditional binding in the full
generator, with matched compute, free image generation, question blanking,
shared-paragraph pairs and independent offline inference. Its training is
running on GPU 0; no new language qualification exists. Finish that comparison
before choosing another architecture or changing the training distribution.

The practical research direction is to learn a compact, prediction-sufficient
visual state and a coherent conditional image writer, and to test both through
actual answers. If a later experiment changes sampling using capability
responses, fix the observable, sampler, budget and independent evaluation
before production. Do not call a training-loss trend a phase transition or
reuse the language of mathematical singularities as evidence of an AI
singularity. Broader Chinese/English instructions, book reading and historical
writing remain the full objective after this bounded conditional test.
