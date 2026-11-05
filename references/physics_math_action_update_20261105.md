# Physics and mathematics: decisions for image-language learning

Primary sources checked on 5 November 2026. This note extends the
[earlier review](physics_math_followup_20261105.md) and separates a promising
algorithmic hypothesis from an established result. None of the reviewed work
establishes an AI singularity or frontier-model capability on one consumer GPU.

## Results with a plausible connection

| Work and date | Established scope | Decision for ILM |
| --- | --- | --- |
| Martin, *A Spectral Renormalization-Group View of Learning*, 30 October 2026, v1 preprint | Spectral learning hypothesis and small optimizer experiments; a fitted exponent near two is not a language-quality certificate. | Consider layer-collapse diagnostics after useful writing. Preserve the current optimizer and avoid adding a spectral target without a matched ablation. |
| Milanfar, *A Lagrangian View of Flow Matching*, revised 5 September 2026, v2 preprint | Target invariance explains straight characteristics; intersecting characteristics limit the single-step ideal. | Measure how the predicted clean writing changes along a sampling trajectory before attempting sampler distillation. |
| Whitelam, *The thermodynamic freedom of a thermodynamic computer*, 28 August 2026, v1 preprint | Simulated 74-unit stochastic computer, MNIST classification, Wasserstein runtime/entropy bounds and operational tradeoffs. | Use transport as a conceptual guide. The bound concerns a Langevin physical device, not CUDA instruction count or measured 4090 energy. |
| Thermodynamic initialization study, published 2 November 2026 | Relaxation of the studied quadratic Langevin systems. | Do not substitute analog relaxation claims for a GPU language-training experiment. |

Primary manuscripts:
[spectral RG](https://arxiv.org/html/2611.00422v1),
[Lagrangian flow](https://arxiv.org/html/2609.00198v2),
[thermodynamic freedom](https://arxiv.org/html/2608.27938v1),
[thermodynamic initialization](https://www.nature.com/articles/s44335-026-00103-2).
These are candidate routes and constraints, rather than four demonstrated
improvements to this model. Preprint and publication dates are distinct.

## The useful mathematical question

Can observed writing pixels retain the information needed to choose an answer,
and can a conditional writer reproduce that choice without destroying strokes?
This requires both semantic selection and faithful synthesis. Removing Unicode
IDs changes the interface; it does not remove the information and optimization
needed to learn a conditional language distribution.

For a fixed prompt image x, let P_x be its true answer-image distribution. Let
R_x be a reference distribution over a continuous state h, K(dy|h) an ideal
writer, Q_x a learned state distribution, and L(dy|h) a learned writer. The
triangle inequality and contraction of total variation under a Markov kernel
give the diagnostic bound

```text
TV(P_x, Q_x L)
 <= TV(P_x, R_x K)
  + TV(R_x, Q_x)
  + E_{h ~ Q_x} TV(K(.|h), L(.|h)).
```

The terms describe representation sufficiency, conditional-state error and
writing error. This is an elementary bound under the specified common latent
space, not a new theorem, measurable training objective or assurance that our
chosen representation is sufficient. A deterministic state is allowed as a
point mass. The first term disappears only if the reference factorization is
correct. Lower flow loss alone does not bound all three terms.

For a finite B-block answer, a sequential coupling further bounds sequence TV
by the sum of conditional block errors (using expectations under the chosen
reference prefix distribution). Thus even modest per-block errors can spoil
long answers. This motivates reporting full-stream correctness alongside
block reconstruction, rather than inferring a working language model from
clean isolated glyphs.

## A direct transport diagnostic

Our convention moves Gaussian noise to data as t increases. For the learned
velocity v and fixed observed condition h, define

```text
x'(t) = v(x(t), t, h)
clean_hat(t) = x(t) + (1-t) v(x(t), t, h).
```

An ideal straight path has a constant clean_hat. A finite-difference measure
of its change along the actual trajectory can reveal where a writer revises
its endpoint. This motivates a future registered diagnostic at fixed weights
and noise, retaining both raw continuous states and thresholded writing.
Small endpoint drift does not establish a correct answer: a consistently
wrong image also has small drift. Quality and question interventions remain
necessary. This diagnostic has not yet been run and changes no current audit.

## Current experiment and next decision

The grounded actor has passed execution checks at 17.181 GiB allocated peak;
that is now measured, superseding the earlier review's unverified execution
status. It inherits externally pretrained vision and contextual weights.
The writing-only stage completed 10,000 updates at 15.801 GiB peak on GPU 0.
Its first evaluation attempt stopped before producing any development output:
the destination directory was missing. The final EMA and training journals
were retained. Recovery creates the directory and resumes unchanged code at
update 10,000, with no further training or threshold changes. The authoritative
component result belongs in its separate result note once the audit finishes.

The recovered audit is now complete: aligned/shuffled foreground F1 is
34.000%/21.420%, with zero exact blocks and both registered gates failed.
The [completed result](grounded_writer_v64_result.md) and raw-image verifier
retain the exact checkpoint and all cases. No further optimizer update was
performed during recovery. The failure strengthens the need to diagnose
writing sufficiency before optimizing sampler speed; it does not establish
which component caused the lost detail.

Proceed by the registered gates: first held-out source-image reconstruction
with aligned versus shuffled visual conditions, then prompt-dependent writing
under a separately registered instruction experiment. Add faster sampling only
after the writer preserves meaningful writing. The present 7.548B actor is an
initialization and execution path, not the final lightweight model. Distilling
a smaller autonomous actor is contingent on actual language capability.

Neither reconstruction nor inherited text-context sanity establishes an
independent usable ILM. The goal remains Chinese/English language generation
from observed images, later extending to historical and unencoded forms,
with complete provenance and no vocabulary-dependent runtime.
