# Dense Visual Future Energy V28: Research Direction

Date: 2026-08-13

Status: design analysis completed before V28 implementation

## Research Question

Can a small image-only student learn a useful distribution over the next
written Chinese image while preserving arbitrary glyph pixels, using one RTX
4090 and no token, Unicode, OCR, vocabulary, or deployed candidate-bank path?

V25-V27 answer a narrower question negatively. They do not reject the ILM
program. They identify three concrete failures to remove:

1. V25 supervised many local transitions, but one normalized future vector
   collapsed a multimodal next-writing distribution and did not beat an image
   unigram or symbolic bigram.
2. V26 represented several future particles, but supervised only a small set
   of endpoint horizons and did not bind changed history to changed targets.
3. V27 optimized the retina, compressed all 64 cells into one normalized
   query, and trained endpoint compatibility. Its exact suffix-pair assignment
   stayed at chance and its natural score did not beat the controls.

The next experiment should combine the strongest non-conflicting parts:
dense causal supervision, multiple continuous futures, arbitrary-image
scoring, a fixed retinal basis, and an explicit intervention on reading order.

## What The Corpus Can Support

A host-only learnability diagnostic was run on the same public-domain manifest,
training/development identifier partition, 1,024-form audit bank, and 2,048 V27
development windows. It used no student features. Add-`0.1` symbolic n-grams
gave:

| Order | Top-1 | Top-5 | Mean target log probability | Context coverage |
|---:|---:|---:|---:|---:|
| 1 | 2.051% | 7.959% | -6.3465 | 100.0% |
| 2 | 12.598% | 29.639% | -5.0200 | 99.95% |
| 3 | 20.264% | 35.400% | -5.1924 | 89.60% |
| 4 | 17.334% | 25.781% | -5.7767 | 56.05% |

Higher unsmoothed orders lose coverage and back off to unigram in this
diagnostic. The important result is not that trigram is a sufficient language
model. It is that the held-out records contain useful ordered signal beyond
the bigram benchmark. V28 therefore tests a learnable relation.

The V28 evaluator will reproduce unigram, bigram, and trigram diagnostics from
training strings. These remain host-only controls and never enter the student
checkpoint or forward path.

## Representation Invariant

The authoritative written-language object remains

\[
X \in [0,1]^{B\times N\times1\times32\times32}.
\]

Each time slice is a continuous glyph image. `N` is reading time. This is the
clean interpretation of the proposed three-dimensional stream: two retinal
axes plus one ordered language axis.

The existing serpentine lattice may fold the stream into a page or latent
field,

\[
\Phi: [0,1]^{N\times1\times32\times32}
\leftrightarrow [0,1]^{1\times32R\times32C},
\]

but `Phi` must be exactly invertible over valid cells. It is a compute view,
not a new language representation. V28 keeps sequence length 64 so that the
causal mechanism can be falsified before adding page-scale geometry. A 2D fold,
depth channel, or movie is unauthorized to repair a failed 64-cell relation.

## Fixed Perception And Learned Semantics

Let the accepted V16 retina be a frozen map

\[
r_i = \operatorname{normalize}(R_0(x_i)) \in \mathbb R^{192}.
\]

A small shared residual adapter learns a semantic visual view without changing
the retinal basis:

\[
z_i = \operatorname{normalize}(A_\theta(r_i)) \in \mathbb R^{192}.
\]

The context token is formed only from `(r_i,z_i)`. An exponential-moving-
average copy `A_bar` encodes target and candidate images. The fixed raw branch
and learned semantic branch are audited separately over the same 1,024-image
candidate set. This removes the mismatched-scope ambiguity found after V27.

Freezing `R_0` is an isolation decision, not a claim that V27 damaged it. The
two preregistered V27 identity values used different candidate-set sizes and
cannot establish a before/after change.

## Causal Future Field

An eight-block width-384 causal rotary field maps the first 64 image cells to a
state at every reading position:

\[
(h_1,\ldots,h_{64}) = F_\theta((r_1,z_1),\ldots,(r_{64},z_{64})).
\]

For future horizons `d in {1,2,4}`, each state emits four weighted continuous
hypotheses:

\[
\{q^{r}_{t,d,k},q^{z}_{t,d,k},\pi_{t,d,k}\}_{k=1}^{4}.
\]

For an arbitrary candidate image `y`, the model computes fixed raw and EMA
semantic keys `(r_y,z_y)`. Its image energy is

\[
e_{t,d,k}(y)=
\tau_r^{-1}\langle q^r_{t,d,k},r_y\rangle+
\tau_z^{-1}\langle q^z_{t,d,k},z_y\rangle,
\]

\[
s_{t,d}(y)=\operatorname{logsumexp}_k
[\log\pi_{t,d,k}+e_{t,d,k}(y)].
\]

This is not a vocabulary softmax. `y` is a floating image tensor supplied at
runtime. A bank is used only by the evaluator to measure retrieval. The four
raw hypotheses are also continuous writer intentions for a later stage.

## Learning Objective

### Stratified dense prediction

Every update selects 16 causal positions: four from each 16-cell quartile, with
position 64 always included. This gives broad local and long-history
supervision without the quadratic cost of flattening every position into one
large contrastive bank. Horizons `1`, `2`, and `4` produce up to 48 supervised
future relations per rendered stream.

For each font direction and horizon, the opposite-font target images at the
selected positions form an in-batch arbitrary-image candidate set. Positives
are determined only by equality of deterministic canonical target pixels. The
student never receives the resulting group labels as an input.

The main loss is position-weighted multi-positive image NCE:

\[
\mathcal L_{\mathrm{dense}}=-\sum_i w_i
\log\frac{\sum_{j\in P_i}\exp s_i(y_j)}
{\sum_j\exp s_i(y_j)}.
\]

Longer contexts receive larger `w_i`, and the endpoint receives an additional
factor of two.

### Candidate-free continuous score

Ranking alone could learn only the temporary training candidate sets. The raw
future hypotheses therefore also receive the empirical energy score

\[
\operatorname{ES}(Q,y)=
\sum_k\pi_k\lVert q_k-y\rVert_2-
\frac12\sum_{k,l}\pi_k\pi_l\lVert q_k-q_l\rVert_2.
\]

This is candidate-free and rewards a distribution of continuous visual futures
rather than a single average glyph. Continuous visual autoregression via
strictly proper scores motivates this route, while V26 shows that the energy
score must be paired with stronger causal and ranking controls rather than used
alone.

### Cross-font image identity

The online adapter must retrieve an independently rendered EMA view of the same
canonical image in both directions. This contrast is computed on the same
selected future images as the language objective. It is a visual invariance
loss, not a character classifier.

### Order intervention

For the natural endpoint, cells 1-60 are permuted while cells 61-64 and every
candidate pixel remain fixed. The correct target log probability under the
ordered context must exceed the shuffled value by a fixed margin.

For exact suffix-4 pairs, define the assignment margin

\[
m(S)=\frac12\sum_{i=1}^{2}
[S_{i,\pi(i)}-S_{i,1-\pi(i)}].
\]

The full ordered context must beat both its suffix-only and prefix-shuffled
versions. This directly trains and audits the relation that V26 and V27 failed.
Candidate columns remain independently permuted, so row position cannot carry
identity.

## Why This Is The Smallest Useful Next Step

V28 does not add a page encoder, OCR teacher, text tokenizer, external LLM,
writer, diffusion stack, historical-glyph specialization, or billion-parameter
scale. Those components cannot answer whether the image stream learns ordered
language.

It adds only what the prior falsifications require:

1. more than one continuous future;
2. many causal positions and future distances per update;
3. an exact order counterfactual in the training objective;
4. candidate scoring that accepts arbitrary pixels; and
5. same-scope perception audits.

If this mechanism beats shuffled context and the symbolic bigram, it becomes a
defensible language state to which the existing continuous writer can later be
attached. If it fails, a larger lattice or writer remains unjustified.

## Falsification Conditions

V28 is not accepted merely because training loss decreases. It is rejected if
any of these central claims fail on development data:

- full context does not beat a suffix-preserving shuffle;
- suffix-pair assignment remains near chance;
- natural image prediction does not beat unigram and bigram controls;
- the learned adapter cannot improve same-scope cross-font retrieval;
- candidate permutation changes the recovered assignment;
- any string, token, Unicode ID, OCR output, glyph lookup, or external LM enters
  the student path; or
- the run exceeds the fixed one-RTX-4090 resource envelope.

The frozen writing partition remains sealed unless the preregistered gates
pass. A writer is trained only after language selection.

## Primary Research Basis

- Rust et al., [Language Modelling with Pixels](https://arxiv.org/abs/2207.06991),
  demonstrates vocabulary-free pixel inputs and cross-script transfer.
- Tai et al., [PIXAR: Auto-Regressive Language Modeling in Pixel
  Space](https://arxiv.org/abs/2401.03321), demonstrates autoregressive pixel
  language generation and documents the readability problem of direct
  maximum-likelihood image generation.
- Gloeckle et al., [Better & Faster Large Language Models via Multi-token
  Prediction](https://arxiv.org/abs/2404.19737), motivates simultaneous future
  horizons for sample efficiency and induction.
- Pang et al., [Next Patch Prediction for Autoregressive Visual
  Generation](https://arxiv.org/abs/2412.15321), shows that grouped visual
  futures can lower autoregressive cost without requiring a new tokenizer.
- Shao et al., [Continuous Visual Autoregressive Generation via Score
  Maximization](https://arxiv.org/abs/2505.07812), provides the proper-score
  basis for continuous, non-quantized future distributions.
- Chen et al., [A Simple Framework for Contrastive Learning of Visual
  Representations](https://arxiv.org/abs/2002.05709), motivates a separate
  nonlinear visual projection and sufficient cross-view contrast.
- Kesen et al., [Multilingual Pretraining for Pixel Language
  Models](https://arxiv.org/abs/2505.21265), provides evidence that pixel models
  can develop cross-script linguistic structure rather than only typography.
