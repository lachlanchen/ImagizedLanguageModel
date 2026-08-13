# Conditional Visual Density-Ratio Field V29: Research Direction

Date: 2026-08-13

Status: design analysis completed before V29 implementation

## Research Question

Can an image-only student learn the *incremental* evidence that earlier written
context contributes to an arbitrary next-glyph image, when the final four
glyph images are held exactly fixed?

V28 learned useful cross-font visual identity and detected some ordered-context
signal, but it did not turn that signal into the correct next-writing choice.
V29 isolates this binding problem. It does not add more parameters, page-scale
geometry, a writer, historical-glyph specialization, or a teacher language
model to conceal an unresolved causal failure.

## Evidence That Forces A Different Objective

The preregistered V28 run produced:

- full-context 1,024-way top-1 `1.4160%`;
- image-unigram top-1 `1.8555%`;
- symbolic-bigram top-1 `13.1348%`;
- full exact-suffix pair arm accuracy `49.5605%`;
- shuffled-prefix pair arm accuracy `49.9512%`;
- raw 1,024-way cross-font identity `92.0410%`; and
- EMA-semantic cross-font identity `96.4355%`.

The model could see candidate identity and could change a scalar target score
when earlier context was shuffled. It nevertheless failed to bind each context
row to its correct candidate column.

Two V28 design choices explain why its loss could improve without solving that
relation:

1. A candidate-independent future vector compressed all possible next images
   before the candidate was known. A dot product then asked that one vector to
   represent incompatible futures.
2. The pair-order loss reduced two row margins to their mean before applying
   its penalty. A large improvement on one row could compensate for a wrong
   assignment on the other row.

### Post-hoc subtraction is not a repair

The frozen V28 checkpoint was re-evaluated on the same 2,048 natural windows
and 512 suffix pairs. No parameters were changed. Subtracting the suffix or
shuffled score after training did not recover binding:

| Score used at evaluation | Natural top-1 | Target log probability | Pair arm accuracy | Pair both-correct |
|---|---:|---:|---:|---:|
| full | `0.014160` | `-6.829707` | `0.495605` | `0.096680` |
| suffix-4 | `0.015625` | `-6.860076` | `0.500000` | `0.000000` |
| shuffled prefix | `0.002441` | `-7.044819` | `0.499512` | `0.020508` |
| full minus suffix | `0.008301` | `-6.957439` | `0.509766` | `0.113281` |
| full minus shuffled | `0.012695` | `-6.892106` | `0.507324` | `0.076172` |

Full and suffix pair margins have correlation `0.68933`; full and shuffled
margins have correlation `-0.05850`. The high first correlation shows that
candidate/suffix evidence dominates the learned score. Arithmetic performed
only by the evaluator cannot create a conditional relation that the training
objective never made identifiable.

## Visual-Time Representation

The authoritative object remains an ordered stream of continuous images:

\[
X\in[0,1]^{B\times N\times1\times32\times32}.
\]

This is the direct interpretation of a `32 x 32 x N` written-language volume:
two retinal axes and one reading-time axis. It does not assume a Unicode code
point, a character class, or even that every cell is a modern character.

The stream can be folded into a page canvas,

\[
\Phi(X)\in[0,1]^{B\times1\times32R\times32C},\qquad RC\geq N,
\]

provided that `Phi` is invertible over valid cells and preserves an explicit
reading-order map. A 2D fold is a rendering or compute view. Arbitrary 2D
adjacency must not silently replace language order. The inverse operation can
turn a generated canvas back into a visual-time stream without OCR.

This representation already permits modern Chinese, seal, bronze, oracle,
handwritten, damaged, or uncoded forms because a cell is pixels rather than an
ID. V29 keeps `N=64` to prove conditional language binding before testing a
larger page or movie representation.

## Conditional Density-Ratio Factorization

Split the context into an earlier prefix `P`, an exact suffix `S`, and an
arbitrary candidate image `Y`. Let `q(Y)` be the candidate/noise distribution
used by a contrastive batch. A sufficiently expressive contrastive critic for
the full context has the density-ratio form

\[
F^*(P,S,Y)=\log\frac{p(Y\mid P,S)}{q(Y)}+a(P,S),
\]

while the same critic applied to the suffix has

\[
B^*(S,Y)=\log\frac{p(Y\mid S)}{q(Y)}+b(S).
\]

Their difference is

\[
G^*(P,S,Y)=F^*(P,S,Y)-B^*(S,Y)
=\log\frac{p(Y\mid P,S)}{p(Y\mid S)}+c(P,S).
\]

The candidate prior and suffix evidence cancel. `G` is therefore the
candidate-specific evidence contributed by the earlier visual history. The
remaining `c(P,S)` is independent of `Y`, so it cancels in row-wise ranking.
For symmetric pair assignment, V29 explicitly row-centers the delta:

\[
\widetilde G_{ij}=G_{ij}-\frac{1}{K}\sum_{k=1}^{K}G_{ik}.
\]

This also removes arbitrary context-dependent offsets before candidate columns
compare rows.

This derivation is a structured density-ratio argument, not a claim that the
loss directly estimates mutual information. Recent analysis distinguishes
contrastive representation learning from accurate mutual-information
estimation. V29 consequently reports predictive and interventional behavior,
not an MI number.

## Candidate-Conditioned Visual Critic

V29 retains the frozen V16 retina and the validated frozen V28 semantic
adapter. It initializes the V28 causal context field but discards V28's
candidate-independent future heads.

For context cells, the causal field produces retained states

\[
H(P,S)=(h_1,\ldots,h_T).
\]

For candidate pixels, the fixed visual encoders produce raw and semantic views
`r(Y)` and `z(Y)`. A learned projection forms a candidate query. Two
candidate-to-context cross-attention layers update that query by inspecting all
retained causal states. A small relation head emits

\[
\rho_\theta(H,Y)\in\mathbb R.
\]

The three scores are then

\[
F_\theta(P,S,Y)=\rho_\theta(H(P,S),Y),
\]

\[
B_\theta(S,Y)=\rho_\theta(H(S),Y),
\]

\[
G_\theta(P,S,Y)=F_\theta(P,S,Y)-B_\theta(S,Y).
\]

The candidate is no longer consulted only after the language state has been
collapsed to one future vector. It queries the visual history directly. The
context states are computed once and candidate queries can be scored in
chunks, so the method remains practical for a 1,024-image audit or training
bank on one RTX 4090.

## Learning Signal

### Natural visual language

Natural 64-cell contexts use a deterministic 1,024-image training bank
rendered from training-partition forms. The bank is a host-side set of floating
images, never a learned vocabulary table, checkpoint buffer, or required
inference input. Exact canonical pixels locate positives outside the student.

The same candidates are scored with full, suffix-4, and suffix-preserving
shuffled contexts. Training applies image NCE to `F`, `B`, and `G`, plus an
ordered-versus-shuffled target log-probability margin. This makes the
incremental random variable part of optimization rather than post-hoc
arithmetic.

### Exact suffix collisions

Each pair has two different prefixes and two different next images, but the
last four context images are pixel-identical. Candidate columns are
independently permuted. For each row, V29 directly penalizes a wrong
correct-versus-other delta. It also applies symmetric row/column assignment to
row-centered `G`, and requires each ordered delta margin to exceed its shuffled
counterpart.

This is conditional negative sampling with a controlled collision: negatives
share the exact observed suffix rather than merely looking similar. Such hard
negatives are useful only because their construction and false-negative risk
are known. Randomly choosing visually close negatives without this semantic
control would not justify the same density-ratio interpretation.

## Why This Is The Smallest Defensible V29

V29 changes one causal mechanism:

1. the arbitrary candidate image attends the retained visual history;
2. full and suffix critics share all parameters;
3. their difference is trained directly;
4. pair losses operate per row before reduction; and
5. row centering removes unidentifiable context offsets.

It deliberately does not add:

- OCR, tokenization, Unicode IDs, glyph IDs, or a visual codebook;
- an external language model or teacher state;
- a billion-parameter backbone;
- page-scale 2D geometry or a depth/movie axis;
- a diffusion writer; or
- a historical-form answer benchmark.

Those are later stages. If the 64-cell student cannot use earlier visual
history to select the right arbitrary image under an exact suffix collision,
larger geometry does not establish that it learned language.

## Alternatives Rejected Before Preregistration

### Evaluator-only score subtraction

Rejected because the frozen V28 diagnostic remains near chance. The delta must
be optimized, not merely computed.

### A larger candidate-independent future head

Rejected because more mixture components do not resolve which history feature
is relevant to a particular candidate.

### Immediate page, 3D, or movie modeling

Deferred because it changes data scale and topology while leaving conditional
binding untested. The ordered visual stream can later be folded into or
rendered from these formats without changing the V29 question.

### Pure pair classification

Rejected because it could solve a two-choice laboratory task without learning
a useful natural next-writing distribution. V29 retains the 1,024-way natural
audit and symbolic unigram/bigram controls.

### Direct pixel reconstruction

Rejected for this stage because glyph rendering loss can dominate weak
language evidence. A writer is authorized only after the critic selects the
right image from context.

## Falsification Conditions

V29 is rejected if any central result occurs:

- incremental pair assignment remains near chance;
- the increment does not beat its suffix-preserving shuffled control;
- the full critic does not beat unigram and bigram controls;
- a candidate permutation changes recovered assignments;
- cross-font candidate identity is insufficient;
- any string, token, Unicode ID, OCR output, lookup table, or external language
  model enters the student path;
- the training image bank becomes a checkpoint or inference dependency; or
- the fixed run exceeds one RTX 4090 and its memory cap.

The frozen partition and writer remain sealed unless every preregistered gate
passes.

## Primary Research Basis

- van den Oord et al., [Representation Learning with Contrastive Predictive
  Coding](https://arxiv.org/abs/1807.03748), motivates future-latent prediction
  with a tractable contrastive density-ratio critic.
- Ceylan and Gutmann, [Conditional Noise-Contrastive Estimation of
  Unnormalised Models](https://arxiv.org/abs/1806.03664), establishes that
  observed data can condition the noise-generation process in NCE.
- Wu et al., [Conditional Negative Sampling for Contrastive Learning of Visual
  Representations](https://arxiv.org/abs/2010.02037), analyzes conditionally
  selected hard visual negatives and their bias/variance tradeoff.
- Ash et al., [Investigating the Role of Negatives in Contrastive
  Representation Learning](https://proceedings.mlr.press/v151/ash22a.html),
  identifies the collision-coverage tradeoff in negative-set size.
- Zheng et al., [Contrastive Difference Predictive
  Coding](https://arxiv.org/abs/2310.20141), motivates learning predictive
  differences rather than relying only on endpoint prediction.
- Ryu et al., [Contrastive Predictive Coding Done Right for Mutual Information
  Estimation](https://arxiv.org/abs/2510.25983), separates accurate MI
  estimation from the structured density ratios useful to representation
  learning.

These works motivate the estimator form. They do not establish that V29 will
learn written Chinese; the fixed experiment must decide that claim.
