# Visual Future Block Language V48: Research Decision

Date: 2026-08-15

Status: mechanism selected after V42--V47 diagnosis and before V48 protocol,
implementation, or training

## Decision

V48 should test one small change to the best measured image-language model in
this repository:

> Learn one causal state from raster history by predicting the next four
> continuous raster fields at every visible position, then draw the next image
> directly from the first predicted field.

The model keeps V42's exact, fixed 32 by 32 DCT image coordinate and its
384-wide, eight-layer causal reader. It removes V42's 8.07M-parameter
stochastic sampler, adds only three learned horizon embeddings to one shared
visual head, and remains about 16.3M parameters. Input, supervision, output,
and recurrent feedback are floating images. A candidate bank exists only in
the evaluator after the model has emitted a raster.

V48 is not a learned tokenizer, an OCR-plus-LLM pipeline, a text model with an
image front end, or a diffusion model that happens to render text. It is a
bounded test of whether finite-horizon visual prediction produces a better
language state than next-image prediction alone.

## Local Evidence That Determines The Design

### V42 is the current positive mechanism

V42 learned next-glyph continuation from one canonical Chinese raster stream:

| Development measure | V42 |
| --- | ---: |
| full-history top-1 | 0.19971 |
| symbolic bigram top-1 | 0.12256 |
| shuffled-history top-1 | 0.18359 |
| ordered-minus-shuffled target log probability | 0.17391 |
| counterfactual arm accuracy | 0.53027 |

Its fixed DCT field is invertible, accepts arbitrary rasters, and has no row per
character. The result is bounded but real language evidence: ordered image
history improves prediction beyond frequency, one-step symbolic context, and
shuffled visual history.

### V47 identifies two paths not to repeat

The matched predictive-state diagnostic shows that V47's learned V34 codec
sphere is inferior to the direct field even before its terminal failure. At 32
visible cells, V47 reaches 0.10498 development top-1 while V42 reaches 0.17627.
V47 then collapses only between 63 and 64 cells, from 0.10498 to 0.02734,
because its independent counterfactual loss uniquely targets the final
absolute position. The same cliff appears on training and development
partitions.

Therefore V48 must not:

- predict V34's raw or normalized reconstruction state;
- add a relation loss only at the final context position;
- use a repeated finite pair table as a substitute for natural language; or
- interpret better reconstruction geometry as better predictive geometry.

### The learned writer is not the bottleneck

The direct-actuator intervention bypassed V42's learned sampler and decoded its
predicted DCT field at the exact zero boundary. On the fixed 2,048-window
development reservoir it obtained:

| Direct field measure | Value |
| --- | ---: |
| proposal identity top-1 | 0.18018 |
| visible-reread identity top-1 | 0.16504 |
| visible pixel F1 | 0.47159 |
| proposal-to-visible cosine | 0.85012 |
| blank rate | 0.00000 |

The registered V42 stochastic sampler obtained 0.08203 identity and 0.37308
F1 on its 256-window generation audit. Although the windows differ and this is
not a gate replacement, the intervention shows that the learned sampler is
both larger and less faithful than exact direct decoding.

A training-only threshold scan also exposes why output quality cannot be
reduced to pixel overlap. The F1-selected threshold improves development F1 to
0.50279 but reduces identity to 0.07959. V48 therefore fixes the inverse-DCT
decision boundary at zero and optimizes language identity and raster quality
jointly.

## Closest Primary Research

### Multi-token prediction

[Better & Faster Large Language Models via Multi-token
Prediction](https://arxiv.org/abs/2404.19737) trains one shared causal trunk
with independent heads for several future positions. It reports improved
sample efficiency, stronger induction on controlled algorithmic tasks, and
faster speculative inference for four-future prediction. This directly
motivates asking one visual state to predict more than the immediate glyph.

Its evidence is not a guarantee for V48. The reported gains become stronger at
larger model sizes, whereas V48 is approximately 16M parameters and predicts
continuous image fields. V48 must therefore preserve a dominant horizon-one
loss and be judged against the measured V42 model, not cite the paper as proof.

### Joint-embedding prediction

[I-JEPA](https://arxiv.org/abs/2301.08243) predicts representations of several
target image blocks from a distributed visual context rather than reconstructing
every pixel. [V-JEPA 2.1](https://arxiv.org/abs/2603.14482) further reports that
dense predictive supervision over visible and masked locations improves
spatial and temporal grounding. These results support two principles:

1. target several informative future regions from one context state; and
2. apply prediction densely, not at one privileged terminal position.

V48 adopts those principles without adopting a learned target encoder. Its
fixed image coordinate already cannot collapse and can be inverted visibly.

### Compact end-to-end predictive models

[LeWorldModel](https://arxiv.org/abs/2603.19312) demonstrates that an
approximately 15M-parameter encoder-predictor can learn useful pixel dynamics
on one GPU in a few hours. It trains a next-embedding objective together with
SIGReg, an isotropic-Gaussian anti-collapse regularizer. This is encouraging
for the compute scale but does not justify adding SIGReg here.

LeWorldModel itself reports weaker performance in its simplest, low-diversity
environment and identifies a mismatch between low intrinsic dimensionality
and a high-dimensional isotropic Gaussian prior as a possible cause. The
canonical-glyph stream is likewise a structured, comparatively
low-intrinsic-dimensional manifold with a large shared background component.
V42 already has an exact non-collapsing field, and V47 showed that spreading
glyph identities apart does not automatically improve language. V48 therefore
does not learn a new encoder and does not impose a Gaussian latent prior.

### Continuous visual generation

[Continuous Visual Autoregressive Generation via Score
Maximization](https://proceedings.mlr.press/v267/shao25b.html) places
codebook-free continuous autoregression under strictly proper scoring rules and
shows that an energy score can train a sample distribution without a tractable
likelihood. [Autoregressive Image Generation without VQ](https://arxiv.org/abs/2406.11838)
uses a diffusion loss to model continuous per-step distributions.

These establish principled routes for a later multimodal visual writer. They
do not make a stochastic head mandatory in the smallest language-state test.
The local intervention says V42's sampler currently destroys a useful visual
proposal. V48 first tests deterministic visible prediction; a proper
distribution head is deferred until the predictive state itself qualifies.

## Predictive-State View

Let `X_t` be the visible 32 by 32 glyph raster and let `K_t = k(X_t)` be its
fixed unit DCT field. A history is `H_t = (K_1, ..., K_t)`. A useful language
state should preserve distinctions between histories that imply different
future writing distributions.

For a four-cell future,

```text
Y_t = (K_(t+1), K_(t+2), K_(t+3), K_(t+4)).
```

The ideal predictive equivalence relation is

```text
H ~ H'  iff  p(Y | H) = p(Y | H').
```

V48 does not estimate this full conditional distribution. It approximates a
finite predictive state by requiring one hidden value `h_t` to support four
visual discrimination problems. A history feature that explains only the
immediate local transition cannot minimize all four losses when later visual
futures differ. Conversely, features stable across several futures receive
gradient from every horizon. This is the intended pressure toward lexical,
syntactic, and semantic regularity; those properties must still be measured
rather than assumed.

## Selected Model

### Fixed sensory and motor coordinate

For binary ink raster `x`, V48 uses V42's exact transform:

```text
s(x) = 2 * 1[x >= 0.5] - 1
k(x) = vec(D s(x) D^T) / 32,
```

where `D` is an orthonormal DCT-II matrix. Because the transform is
orthonormal and every binary signed image has norm 32, `k(x)` is already unit
length. The visible inverse is

```text
x_hat(q) = 1[D^T unvec(32 q) D >= 0].
```

There is no quantizer, nearest-neighbour lookup, glyph inventory, or learned
decoder.

### Shared causal state and horizon-conditioned head

The causal reader is

```text
h_1:T = Transformer(k(X_1), ..., k(X_T)).
```

One shared MLP predicts all horizons. Horizon one uses the unmodified hidden
state. Horizons two through four add one learned 384-dimensional visual-time
offset each:

```text
u_(t,1) = h_t
u_(t,r) = h_t + e_r                         for r in {2,3,4}
q_(t,r) = normalize(W_2 SiLU(W_1 u_(t,r))).
```

This adds 1,152 parameters rather than four 1,024-way output matrices. Sharing
the head forces the horizons to inhabit the same visible image coordinate;
the offsets identify future distance without becoming glyph embeddings.

### Dense visual-future objective

Each training segment contains 68 raster cells. Cells 1--64 are the visible
causal stream. At every position `t` in that stream, horizon `r` is supervised
by cell `t+r`. Thus all 64 absolute positions receive every horizon loss and no
loss uniquely targets position 64.

For each horizon, exact-raster multi-positive contrast compares predicted
fields with target fields dynamically present in the minibatch:

```text
L_r = L_contrast(q_(.,r), K_(.+r))
      + 0.25 * mean(1 - q_(t,r)^T K_(t+r))
      + 0.20 * L_visible-pixel(q_(.,r), X_(.+r)).
```

The fixed horizon weights are

```text
alpha = (1.000, 0.500, 0.250, 0.125),
L_V48 = sum_r alpha_r L_r.
```

The unnormalized first term preserves V42's full horizon-one pressure; later
horizons are auxiliary and geometrically decayed because their uncertainty
grows. This weighting is selected before V48 training and cannot be tuned from
its development result.

The loss does not claim to model the full multimodal future distribution. It
tests predictive-state learning. If the state qualifies but deterministic
generation remains limited, a block energy-score head becomes a separately
preregistered follow-up.

### Inference

For ordinary autoregressive generation, V48 uses only `q_(t,1)`:

1. read up to 64 visible raster cells;
2. predict four continuous future fields;
3. inverse-transform and zero-threshold the first field;
4. append that visible raster; and
5. reread it for the next step.

The four-field forecast may also be rendered as an evaluator-visible planning
strip, but cells two through four are not silently inserted into the recurrent
stream. Replanning after every emitted image avoids blockwise error
accumulation while retaining multi-horizon training.

## Alternatives Rejected For V48

| Alternative | Reason not selected |
| --- | --- |
| retrain V47 without pair loss | removes the terminal cliff but leaves the pre-cliff codec state far below V42 |
| learned JEPA encoder plus SIGReg | repeats a learned target-coordinate question before the exact field is exhausted and risks a low-dimensional prior mismatch |
| larger transformer | does not isolate the objective change and weakens the one-GPU efficiency claim |
| diffusion or flow writer | local evidence says the writer is currently harmful; generation is not the next bottleneck |
| a 1,024-glyph classifier | would make the deployed model inventory-bound and unable to express unseen forms |
| explicit pair curriculum | V43, V44, and V47 did not generalize this intervention reliably and V47 poisoned one absolute position |
| page patches or multiple fonts | introduce segmentation and form variation before canonical visual language is stable |

## Falsifiable Expectations

If the selected mechanism is right, V48 should:

- retain or improve V42 horizon-one continuation;
- increase ordered-minus-shuffled evidence without a special pair loss;
- predict horizons two through four above image-frequency controls;
- remain stable from context length 63 to 64;
- preserve most proposal identity after direct visible rendering;
- train in less memory and with fewer parameters than V42; and
- fail cleanly, without opening the frozen partition, if these effects do not
  occur.

If horizon-one performance falls while later heads learn only corpus
frequency, multi-horizon supervision is competing with rather than enriching
the language state. If all horizons become similar, the horizon-conditioned
head is under-separated. If order sensitivity does not improve, the next
experiment should investigate explicit predictive-state factorization, not add
a more elaborate writer.

## Claim Boundary

A qualifying V48 would show that dense finite-horizon visual prediction is a
better small-model objective for canonical Chinese raster continuation and
that direct continuous-field output can remove a learned glyph generator. It
would still not prove general semantics, instruction following, etymology,
historical-form generation, page-level reading, human-equivalent vision,
Qwen parity, or better scaling than token LLMs.
