# Recent physics and mathematics for a single-GPU imagized language model

Evidence reviewed through 5 November 2026. The practical objective is readable,
question-dependent image-language generation on one RTX 4090, with independent
inference. None of the work below establishes an AI singularity or removes the
need for language data, semantic learning and held-out evaluation.

## The measured problem

V61 trains all 477,145,984 native pixel-language parameters for 1,000 updates
on one RTX 4090 D. Training takes 1,501.1 s through the final optimizer update,
excluding loading and final serialization, with 9.912 GiB peak allocated
memory. The offline deployment bundle reproduces output tensors, PNG bytes
and stopping events with empty source/model caches and networking disabled.

The language test fails: 0/256 exact answers, 0/393 pairs both correct, and
0.149% exact-raster multiset F1. Held-out answer-pixel NLL improves from 0.6842
to 0.1548 nats, yet generated foreground F1 is only 12.264%. A blinded CPU
transcription diagnostic gives no exact model answers; its calibration also
recognizes only 40/256 reference images exactly, so it is an unreliable
semantic instrument for these rasters. These measurements justify changing
the output distribution and checking the font interface before adding scale.

## Recent results and their actual scope

| Work | Established result or author-reported evidence | Relevance to ILM |
| --- | --- | --- |
| **MGFlow**, first posted 28 Sep, latest revision 2 Oct 2026, preprint | Mixture representations, mass-constrained assignment and paired distributional updates address mode collapse in one-step visual generation. Its reported timing benchmark uses eight H200s. | Preserve distinct writing modes. Investigate a small conditional rendering head; the full published training recipe is not a one-4090 result. |
| **W-Flow**, first posted 12 May, latest revision 26 May 2026, preprint and official code | A neural generator learns updates derived from a Sinkhorn/Wasserstein gradient flow, enabling one-evaluation image sampling. Large released recipes use multiple nodes and pretrained feature encoders. | A possible fast glyph renderer after learning an appropriate conditional distribution. One-step inference does not mean cheap foundation training. |
| **Generate in Reconstruction Space, Match in Semantic Space (SemOT)**, revised 2 Nov 2026; author record reports NeurIPS acceptance | Frozen semantic features make finite-batch transport matching more stable in the reported ImageNet experiments; optimizing the evaluation encoder can instead damage sample quality. | Investigate writing-sensitive features for training, while measuring raw strokes and question-specific correctness independently. Natural-image semantic features are not established language features. |
| **Scalable generative sampling and multilevel estimation for lattice field theories near criticality**, published 26 Oct 2026; April preprint | A renormalization-inspired hierarchy combines conditional Gaussian mixtures and flows, preserving coarse fields during refinement in scalar lattice-field experiments. | A possible hierarchy for writing structure and fine strokes. The measured acceleration concerns physical-field sampling, not language learning or GPU foundation training. |
| **Designing Optimal Transport Flows**, 2 Jun 2026, v1 preprint | Designs a low-frequency prior to simplify transport toward natural images, reducing trajectory curvature in reported experiments. | Test coherent coarse writing before fine strokes. Low-frequency agreement alone cannot distinguish similar Chinese characters. |
| **The Expressivity Boundary of Probabilistic Circuits**, 13 May 2026, preprint | Identifies probability-space output and fixed-routing context bottlenecks; its separation-rank results describe worst-case expressivity, with controlled-topology experiments. | Keep flexible contextual reasoning in the transformer. Use a tractable circuit only for local image rendering, and test its limitations. |
| **Digitally optimized initializations for fast thermodynamic computing**, 2 Nov 2026, peer-reviewed | Mpemba-inspired initial covariances suppress slow relaxation modes in quadratic Langevin systems; matrix inversion and determinant protocols are analyzed and simulated. | A possible warm-start principle for an iterative renderer. Its theorem is not a theorem about neural-language training, and physical hardware gains do not transfer directly to a GPU. |
| **Online local learning for generative thermodynamic computing**, 14 Sep 2026, preprint | Local residual-state updates train Langevin systems in digital MNIST-prototype experiments. Heat decreases in the reported paired runs, while diversity decreases too. | A research option for local visual adaptation. General language learning and continual book understanding remain untested. |

Primary sources:

- [MGFlow, latest version reviewed](https://arxiv.org/html/2609.35763v4), especially Sections 3 and Appendix I.
- [W-Flow, latest version reviewed](https://arxiv.org/html/2605.11755v2) and [official implementation](https://github.com/hanjq17/W-Flow).
- [SemOT, 2 November revision](https://arxiv.org/html/2606.00514v2) and [author-maintained implementation](https://github.com/Genentech/semantic-transport-generation).
- [Multilevel lattice-field sampling, published article](https://journals.aps.org/prd/abstract/10.1103/jjss-lwzv) and [April preprint](https://arxiv.org/abs/2604.10209).
- [Designing Optimal Transport Flows, versioned paper](https://arxiv.org/html/2606.04092v1).
- [Probabilistic-circuit expressivity, versioned paper](https://arxiv.org/html/2605.12940v1).
- [Mpemba initialization, published article](https://www.nature.com/articles/s44335-026-00103-2) and [March preprint](https://arxiv.org/html/2603.24183v1). The publication is new; the preprint predates it.
- [Online local thermodynamic learning](https://arxiv.org/abs/2609.15439).

The related peer-reviewed [Generative Thermodynamic Computing](https://journals.aps.org/prl/abstract/10.1103/kwyy-1xln)
(20 January 2026) demonstrates a generative Langevin framework in digital
simulation and proposes analog realization. It motivates structured stochastic
dynamics, but does not demonstrate language understanding or establish an
energy advantage for software simulation on our 4090.

## The mathematical change worth testing first

An independent-pixel decoder has the law

    p(y | h) = product_i Bernoulli(y_i; sigmoid(a_i(h))).

Its BCE optimum estimates conditional pixel marginals. If two possible writing
images have bit patterns 1100 and 0011 with equal probability, all four
marginals are 0.5. Uniform thresholding can return 0000 or 1111, although
neither is a possible correct image. Good pixel marginals therefore do not
guarantee a coherent form. This counterexample is elementary; it is not a
new result or proof that it is the sole cause of V61's failure.

A small, explicit alternative is a conditional mixture of whole-image laws:

    p(y | h) = sum_k pi_k(h) product_i Bernoulli(y_i; sigmoid(a_ki(h))).

Each expert predicts a complete physical image block. Marginalizing its latent
expert creates correlations among pixels. Normalization follows immediately:
each product distribution sums to one, and the mixture weights sum to one.
The exact likelihood costs O(KP) for P pixels and K experts. It introduces no
character labels, Unicode IDs, stored glyph images or fixed writing alphabet;
every expert's output varies with the language context.

For a 896-dimensional state and 2,048-pixel native block, eight linear heads
and a gate add approximately 12.85M parameters over the existing single head,
about 2.7% of the complete model. Five FP32 parameter/state arrays would add
about 0.24 GiB before activations. This is a capacity/memory calculation,
not a measured training result or a language-quality guarantee.

Inference can construct each expert's thresholded image and score every
candidate under the complete mixture. This is an approximate mode search,
not guaranteed global MAP inference. Mixture collapse, blurred experts and
incorrect semantic selection remain possible and must be measured.

This proposal borrows the general lesson of preserving distributional modes;
it does not reproduce MGFlow or inherit its results. A probability mixture
alone is not sufficient: the expressivity paper cautions against replacing
contextual reasoning with fixed probabilistic routing. Conditional image
heads also differ from mixtures over fixed vocabulary distributions.

## Experiment order fixed before the matched test

1. Retain V61's failed decision and raw outputs. Preserve the distinction
   between fitting pixels and generating meaningful writing.
2. Audit rendering with the released model's actual bundled font interface.
   Compare complete prompts and answers without truncation; canonical Noto
   26-pixel cells are a documented shift from that interface.
3. Register a matched single-head versus correlated-head experiment using the
   same foundation, pixels, data choices and update budget. Test normalized
   likelihood, finite full-backbone gradients, expert use, raw glyph fidelity,
   question controls, paired answers and actual single-GPU resources.
4. If that component works, test the prepared broad instruction corpus with
   a genuinely unscored final split. Distribution matching must be conditional:
   reproducing the marginal distribution of written answers can still ignore
   the question entirely.
5. Consider OT-based or warm-start refinement only when it solves a measured
   residual problem. Physical analog-computing claims require separate hardware
   evidence; they are not training-budget reductions for our current machine.

The priority was a coherent conditional distribution over writing images,
coupled to useful language context. That implementation now fits the measured
memory budget; its completed test below rejects the language intervention.

## Semantic geometry without losing the question

The 2 November SemOT revision makes the choice of training geometry especially
relevant. An image feature used for writing must retain distinctions such as an
extra stroke, character order and negation, while tolerating harmless changes
in font or position. This is an ILM requirement, not a demonstrated property of
the paper's natural-image features. Its feature-training recipes also use large
global batches and pretrained teachers; their full cost cannot be assigned to
our single-GPU adaptation budget.

There is a simple failure case for importing an unconditional transport loss.
Let two equally likely questions have different correct answer images A and B.
A generator that ignores the question and randomly emits A or B has exactly the
correct marginal answer distribution. Any distance between the two marginal
distributions is zero, but its expected question-specific accuracy is only
one half. Swapping the answers deterministically gives the same marginal and
zero accuracy. Thus marginal visual distribution matching cannot establish
language understanding.

A conditional transport experiment would need matching restricted to the same
prompt, with multiple legitimate answers or writing variants supplied during
training. A cheaper paired feature loss is also possible when only one answer
is available. Either would supplement a physical-image likelihood; neither
replaces question controls. Frozen teacher features could guide training and
be discarded for actor inference, with their provenance and preparation cost
reported separately. Features and hyperparameters should be selected on train
data before evaluating the development questions.

The initial operational decision was to compare the whole-image density with
a single-head control and audit the font interface first. That test is now
complete. Conditional feature supervision and learning on actual generated
trajectories are candidates for a new experiment because question-dependent
writing still fails. Coarse-to-fine refinement is a further option, provided
it retains small strokes rather than treating them as noise.

## Implementation and falsification

The feasibility prototype led to a completed matched test with 489,998,208
parameters for K=8 and 477,145,984 for K=1. Each makes 1,000 full-backbone
updates on identical sampled examples, sequentially on GPU 0. Held-out pixel
NLL improves from 0.1542 to 0.1248, but both actors have zero exact primary
answers and zero correct question pairs. The eight-expert question effect
does not resolve zero, and the registered intervention is rejected. This
does not falsify mixtures generally; it shows that this density change alone
does not solve conditional writing at the tested budget. The
[complete result](correlated_native_density_v62_result.md) retains confidence
intervals, raw images, resources, replay and failed gates.

The closely related [MIXAR revision of 15 October](https://arxiv.org/html/2604.11575v2)
also distinguishes supervised pixel fit from free-running generation. Its
second stage uses rollout reinforcement learning with reconstruction and
reference anchoring; its evaluation restricts what a vision judge can see
because visible questions permit answer guessing. These are relevant design
lessons, not evidence that copying its recipe onto our shifted interface
will succeed. A new rollout or semantic-feature objective needs its own
registered comparison and independent actor, rather than a changed gate for
the completed experiment.

The active goal remains an independently deployed, useful image-language
model. None of these papers or local experiments establishes a singularity,
frontier-model equivalence, or a way to omit semantic training data.

A [follow-up on recent alignment and latent reasoning](recent_alignment_and_latent_reasoning_20261105.md)
reviews PixelDense, LARC and Visual Prompt Engineering against this deployment
boundary. It separates their transferable training ideas from retained text
encoders/tokens and multi-GPU recipes, and outlines a conditional-feature
hypothesis for a new registered experiment. No new language pass is claimed.
