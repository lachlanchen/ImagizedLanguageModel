# Spatial Visual Next-Field V30: Research Direction

Date: 2026-08-13

Status: design analysis completed before V30 protocol, implementation, smoke
training, or development measurement

## Research Question

Can an image-only causal student predict a spatial continuous field for the
next written image, such that aligned local evidence binds an arbitrary
candidate image to the history that licenses it and beats an equally sized
position-blind visual control?

V29 answered a narrower question negatively. Its candidate-conditioned scalar
critic learned a large natural ordered-context effect but stayed at chance on
pixel-identical-suffix assignment. V30 must change the learned random variable,
not only the baseline or score head. It should emit a candidate-independent
next-image field that can later drive a writer, while candidate images are used
only to train and audit that field.

## Evidence That Forces A Spatial Predictive Target

The fixed V29 development result is:

- full 1,024-way top-1 `2.3438%`;
- symbolic-bigram top-1 `13.8672%`;
- full target log probability `3.102479` nat above suffix-preserving shuffle;
- full exact-suffix pair arm accuracy `49.7314%`;
- incremental pair arm accuracy `50.7080%`;
- incremental both-correct `8.9844%`;
- raw two-candidate identity `99.9512%`; and
- frozen-semantic 1,024-way cross-font identity `96.4355%`.

The scalar critic can see the candidate and can detect that a prefix has been
shuffled, but it does not learn which candidate belongs to which prefix. Its
pair loss also fails on its training distribution: over the last 1,000 logged
updates, mean incremental arm accuracy is `51.5050%`, both-correct is
`11.1306%`, and pair loss is `0.700789`. This is not merely a held-out
generalization gap. The scalar candidate/history interaction is a poor
optimization substrate for the required relation.

V29 also proves that suffix subtraction cannot repair this critic. For an exact
shared suffix, the baseline cancels algebraically from aggregate two-by-two
assignment margin. V30 therefore removes density-ratio subtraction from the
primary mechanism. Suffix-only and suffix-preserving shuffle remain controls.

## Frozen-Retina Spatial Probe

The V16/V29 retina already exposes its continuous pre-pooling field. For a
`32 x 32` image, this field has shape `192 x 4 x 4`. A design-only probe used
the fixed V29 checkpoint, the same 1,024 evaluator forms, and the two
development fonts. It changed no parameter and inspected no language context.

| Frozen visual representation | Bidirectional cross-font top-1 |
|---|---:|
| pooled raw-retina vector | `0.921875` |
| flattened `192 x 4 x 4` field | `0.963379` |
| aligned, channel-normalized `4 x 4` cells | `0.963379` |
| aligned cells after per-image spatial centering | `0.969727` |
| position-blind mean of normalized cells | `0.143066` |
| aligned cells with the second view spatially reversed | `0.003418` |
| raw aligned `8 x 8` pixel patches | `0.896973` |

For the correct cross-font identity, aligned-field cosine exceeds reversed-field
cosine by `0.738075` on average. The field therefore contains strong
font-robust identity specifically in corresponding retinal locations. Its
`96.34%` identity is also closely matched to V29's frozen global semantic
identity of `96.44%`. This supports a controlled local-versus-global test.

The probe is design evidence, not a V30 language result. It was measured before
the V30 protocol and cannot select a V30 checkpoint or threshold. The sealed
frozen partition remains unopened.

### Representation decisions from the probe

V30 should use channel-normalized raw retinal cells. It should not use the
`98.88%` variant obtained by subtracting an evaluator-bank population mean,
because that would make the model depend on development-derived bank
statistics. Per-image spatial centering is deployment-valid, but it removes a
global component needed for a clean matched control; it remains an audit
ablation rather than the primary path.

Raw `8 x 8` pixels underperform the learned retinal field. A smaller thumbnail
would discard still more stroke topology. The existing frozen field is both
more accurate and cheaper than introducing a new visual tokenizer.

## Model-Facing Visual Language Object

The authoritative input remains an ordered stream of continuous writing:

\[
X\in[0,1]^{B\times T\times1\times32\times32},\qquad 1\leq T\leq64.
\]

Reading time is the second axis. A `32 x 32 x T` volume, a serpentine page
fold, or a writing movie is a reversible view of this stream only when its
reading-order map is explicit. V30 does not introduce a hidden character axis,
Unicode channel, or visual codebook.

The new model output is a continuous next-image field:

\[
P_\theta(X)\in\mathbb R^{B\times16\times192}.
\]

The 16 rows retain the `4 x 4` retinal topology. This is a useful generative
state even when no candidate bank exists. A later continuous motor model can
render it to pixels; V30 itself tests prediction and binding before authorizing
that writer.

## Spatial Next-Field Predictor

V30 initializes the frozen retina, frozen semantic adapters, and trainable
causal context field from the fixed V29 checkpoint. The V29 candidate query,
evidence blocks, relation head, and suffix baseline are discarded.

Let the retained causal states be

\[
H_T=C_\theta([r(x_1);z(x_1)],\ldots,[r(x_T);z(x_T)])
\in\mathbb R^{T\times384}.
\]

A compact field decoder maps the final causal state to 16 seeded local states,
mixes neighboring cells with a small residual `4 x 4` convolutional field, and
projects each cell to 192 dimensions:

\[
Q_0=\operatorname{reshape}_{4\times4}
  (W_{seed}h_T+b_{seed}),
\]

\[
Q_{l+1}=Q_l+\operatorname{ConvFieldBlock}_l(Q_l),\qquad
P_\theta(X)=\operatorname{normalize}_{c}(W_{out}Q_L).
\]

The prediction is candidate-independent. This is important: deployed next-state
inference consumes context images and emits a continuous visual state directly.
Candidate images enter only a contrastive scorer used for training or
evaluation.

An attention decoder remains a plausible later alternative, but a convolutional
`4 x 4` field is the smaller and more attributable first test. The causal
backbone has already integrated temporal context; V30 asks whether decoding
that state into spatial visual evidence fixes binding.

## Frozen Candidate Fields

For candidate image `y`, the frozen retina produces

\[
U(y)\in\mathbb R^{16\times192},\qquad
u_p(y)=\frac{U_p(y)}{\lVert U_p(y)\rVert_2}.
\]

The spatial score compares corresponding locations before reducing to a scalar:

\[
s_{sp}(X,y)=\tau\frac1{16}\sum_{p=1}^{16}
P_{\theta,p}(X)^\top u_p(y).
\]

This is not a glyph classifier. `y` may be any floating image accepted by the
retina. There is no candidate-ID embedding, vocabulary-sized output, or
required bank.

## Exact Position-Blind Control

The control must retain target visibility and trainable capacity. Simply
averaging the raw local cells is inadequate: the probe shows only `14.31%`
cross-font identity, which would confound topology with perception quality.

Instead, the matched control uses the V29 frozen semantic vector
`z(y) in R^192`, whose same-scope identity is `96.44%`, and tiles it across the
same 16 positions. An independently trained model with the identical field
decoder, parameter count, initialization, batches, optimizer, and update count
uses

\[
s_{gl}(X,y)=\tau\left(\frac1{16}\sum_pP_{\theta,p}(X)\right)^\top z(y).
\]

All 16 predicted rows and every trainable parameter remain available; only the
candidate's spatial correspondence is removed. This makes the two arms
capacity-matched while keeping candidate visibility closely matched.

The arms must be separate checkpoints. A single jointly trained model would
allow one objective to shape the representation used by the other and would
not establish that spatial routing caused any gain.

## Learning Theory

### Dense regression supplies per-location gradients

For a positive target image `Y`, the local feature loss is

\[
L_{field}=\frac1{16}\sum_p
\left[1-P_{\theta,p}(X)^\top u_p(Y)\right].
\]

Unlike V29's one scalar per candidate, this supplies 16 aligned prediction
errors for every example. It is a JEPA-style latent prediction objective: the
student predicts target visual features, not pixels, labels, or discrete
codes.

With squared error, the population optimum is the conditional mean
`E[U(Y)|X]`. For normalized candidate fields, its dot product with `U(y)` is a
kernel-smoothed conditional score. This gives a direct route from continuous
field prediction to candidate ranking. It may blur genuinely multimodal
futures, so dense regression cannot stand alone.

### Contrastive ranking preserves multimodality

For an image candidate set sampled from `q(y)`, row-wise cross entropy over
`s(X,y)` has the usual contrastive density-ratio optimum, up to critic
capacity:

\[
s^*(X,y)=\log\frac{p(y\mid X)}{q(y)}+a(X).
\]

The 1,024-way image contrast prevents a field that merely predicts average
stroke occupancy from being accepted. Exact-suffix two-by-two assignment adds
hard negatives with identical recent visual history. Dense regression and
contrastive ranking therefore address complementary failures: the former
improves optimization, while the latter preserves conditional choice.

### Spatial intervention tests the claimed mechanism

For a fixed nonidentity permutation `pi` of the 16 candidate locations,

\[
s_{perm}(X,y)=\tau\frac1{16}\sum_p
P_{\theta,p}(X)^\top u_{\pi(p)}(y).
\]

A spatially grounded model should lose assignment accuracy and target log
probability under this intervention. The tiled-global control is exactly
invariant. Candidate-column permutations must remain equivariant in both arms;
otherwise the model has learned an index convention rather than an image
relation.

## Fixed Training Signals To Preregister

Both arms should use the same natural windows and exact-suffix pair
specifications as V29, with new fixed seeds established before any V30 run.

The spatial arm should combine:

1. 1,024-way full-context image cross entropy;
2. positive local-field cosine loss;
3. full-context target-log-probability advantage over a suffix-preserving
   shuffled prefix;
4. symmetric two-by-two exact-suffix assignment;
5. per-row correct-versus-other margin; and
6. per-row ordered-versus-shuffled margin.

The global arm uses the identical losses after replacing the local candidate
field with the repeated frozen global semantic vector. Suffix-only scores are
measured but not subtracted or optimized as a primary target.

Training and evaluation should average both cross-font directions. Candidate
columns are independently permuted. The image bank may be cached outside the
model, but it must not enter checkpoints or deployed inference.

## Why This Is The Smallest Defensible V30

V30 changes one mechanism: the model predicts a spatial next-image random
variable before candidate scoring. It does not add:

- tokenization, bytes, Unicode IDs, character IDs, OCR, or a codebook;
- a vocabulary embedding or vocabulary-sized classifier;
- an external language model, teacher state, or symbolic loss;
- a larger corpus, longer context, page fold, depth axis, or movie model;
- a writer or pixel diffusion process; or
- a historical-glyph answer benchmark.

The model remains a compact continuous visual predictor on one RTX 4090. If
aligned fields cannot beat the matched global arm and symbolic bigram, then
page geometry or a writer would only conceal an unresolved language-state
failure.

## Alternatives Rejected Before Preregistration

### Another scalar candidate cross-attention critic

Rejected because V29 already trained that relation for 8,000 updates and
failed both training and held-out pair assignment.

### Better suffix subtraction

Rejected because a shared suffix vector cancels exactly from aggregate pair
margin. A new baseline cannot create missing context-candidate interaction.

### Raw pixel reconstruction as the only target

Rejected because font, antialiasing, and stroke width can dominate the weak
language signal. The frozen retina field is more cross-font identifiable than
raw aligned patches (`96.34%` versus `89.70%`). Pixel generation remains a
later motor problem.

### Development-bank population centering

Rejected despite its `98.88%` identity because it would adapt the student to
an evaluator-derived bank statistic. V30 must operate on an arbitrary image
without that bank.

### Immediate mixture diffusion

Deferred. Continuous diffusion can represent multimodal image futures, but it
adds a stochastic writer and much harder attribution. The deterministic field
must first show conditional ranking above matched controls.

### Immediate 2D page or 3D movie training

Deferred. These are valid future interfaces, but reshaping does not solve
candidate binding. The `T x 1 x 32 x 32` stream already has two retinal axes
and reading time; V30 tests the core predictive relation first.

## Falsification Conditions

V30 should be rejected if any central result occurs:

- spatial exact-suffix pair assignment remains near chance;
- spatial assignment does not beat its independently trained global control;
- ordered context does not beat suffix-preserving shuffle;
- local-patch permutation does not reduce the spatial result;
- candidate-column permutation changes recovered assignments;
- the natural distribution does not beat image unigram and symbolic bigram;
- the frozen spatial field fails its cross-font visibility gate;
- either checkpoint contains the training/evaluator bank or labels;
- any token, Unicode ID, OCR transcript, lookup, or external model enters the
  learned path; or
- either fixed arm exceeds the single-RTX-4090 resource cap.

No frozen image or writer may be authorized unless every preregistered
mechanism and language gate passes in both the evidence and matched-control
comparison.

## Primary Research Basis

- Assran et al., [Self-Supervised Learning from Images with a Joint-Embedding
  Predictive Architecture](https://arxiv.org/abs/2301.08243), predicts target
  block representations from visible image context instead of reconstructing
  pixels.
- Bardes et al., [Revisiting Feature Prediction for Learning Visual
  Representations from Video](https://arxiv.org/abs/2404.08471), shows that
  feature prediction can learn appearance and temporal structure without text,
  labels, negative examples, or pixel reconstruction.
- Mur-Labadia et al., [V-JEPA 2.1: Unlocking Dense Features in Video
  Self-Supervised Learning](https://arxiv.org/abs/2603.14482), motivates dense
  predictive loss for explicit spatial and temporal grounding.
- Bardes et al., [VICRegL: Self-Supervised Learning of Local Visual
  Features](https://arxiv.org/abs/2210.01571), demonstrates that local and
  global visual objectives carry different information and that spatially
  corresponding features can be aligned across views.
- He et al., [Masked Autoencoders Are Scalable Vision
  Learners](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html),
  supports an asymmetric design in which the predictive decoder is much
  smaller than the visual encoder.
- van den Oord et al., [Representation Learning with Contrastive Predictive
  Coding](https://arxiv.org/abs/1807.03748), supplies the contrastive
  future-prediction and density-ratio basis for candidate ranking.
- Li et al., [Autoregressive Image Generation without Vector
  Quantization](https://arxiv.org/abs/2406.11838), establishes that continuous
  visual values can support autoregressive generation without a discrete
  visual tokenizer. V30 tests the deterministic field needed before adding
  such a stochastic continuous writer.

These papers motivate the factorization. None proves that V30 learns written
Chinese; the preregistered matched experiment must decide that claim.
