# Codec-Spherical Glyph Language V47: Research Decision

Date: 2026-08-15

Status: design analysis completed before V47 implementation or training

## Decision

The next useful experiment is not another larger raw-pixel predictor and not a
repeat of V35's reconstruction-latent regression. V47 should combine the
strongest two mechanisms already measured in this repository:

1. V34's frozen, codebook-free visual retina and actuator; and
2. V42's from-scratch causal learning over one visible Chinese glyph cell per
   visual time step.

The V34 latent is placed on a unit hypersphere, the causal reader is trained on
natural raster continuations plus a nonrepeating train-only counterfactual
curriculum, and generated states are accepted only through their decoded and
reread pixels. This tests whether a learned glyph manifold is a better
continuous language coordinate than V42's raw DCT sphere or V46's unnormalized
1,024-dimensional retinal field.

## What The Existing Evidence Says

| Result | Relevant observation |
| --- | --- |
| V34 | A 7.42M-parameter continuous codec reconstructs held-font and held-family writing at about `0.998` ink F1 without quantization or a vocabulary. |
| V35 | Predicting V34's raw 768-dimensional reconstruction latent over arbitrary line patches failed language binding; good reconstruction alone is not a language coordinate. |
| V42 | A 24.35M-parameter causal reader trained on isolated glyph rasters reaches `0.19971` held-out next-image top-1 and beats unigram, bigram, and shuffled-history controls. |
| V43 | A spatial writer reaches `0.88240` pixel F1 when given the evaluator's exact visual plan, but only `0.46110` with the autonomous plan. The motor is not the primary bottleneck. |
| V44 | A 24,000-pair tangent correction removes the seen/unseen-train gap but drifts toward the corpus-common field and does not improve development binding. |
| V45 | A fixed centered matrix-power field improves target geometry and passes all representation gates. |
| V46 | Training from scratch in that full field preserves bounded language evidence but passes only 10/14 gates. Its reader does not jointly control direction, radius, and raster quality. |

### V46 radial intervention

A post-result diagnostic used the already-open 256-window development subset
and the frozen V46 checkpoint. It did not alter V46's gate decision.

| Measurement | Value |
| --- | ---: |
| mean predicted anchor radius | `0.21612` |
| mean target radius | `0.98595` |
| predicted/target radius correlation | `0.09433` |
| direct F1, predicted direction and radius | `0.25438` |
| direct F1, predicted direction and one global target radius | `0.42059` |
| direct F1, predicted direction and evaluator target radius | `0.42219` |
| direct F1, target direction and predicted radius | `0.79579` |

The scalar radius is badly collapsed, but target radius barely improves over
one global constant. Correct direction with the bad radius is already much
more readable. Therefore explicit radial calibration is necessary for an
unnormalized field, but visual content direction is the dominant error.

### V34 latent identity intervention

The frozen V34 EMA encoder was evaluated on the 1,024 canonical evaluation
glyphs. Queries were rendered in fonts never used for the V34 training-font
partition and matched by cosine to canonical Noto Sans CJK Regular image
latents. The evaluator used the bank only after encoding.

| Query font | paired cosine | top-1 | top-5 |
| --- | ---: | ---: | ---: |
| Noto Sans CJK Bold | `0.82804` | `0.97754` | `0.99414` |
| Noto Serif CJK Medium | `0.74870` | `0.93750` | `0.98145` |

Thus V34's learned visual state already retains glyph identity across held
fonts. V47 does not need a new style encoder to test canonical language. It
needs to normalize this continuous state, make the reader predict it from
ordered image history, and constrain generation to states that survive the
visible decoder--encoder loop.

## External Evidence

- [Hyperspherical Latents Improve Continuous-Token Autoregressive
  Generation](https://arxiv.org/abs/2509.24335) reports that fixing latent
  radius removes a scale degree of freedom that destabilizes continuous
  autoregressive decoding. This agrees with the V46 radial collapse, although
  its natural-image results do not prove glyph language.
- [Continuous Visual Autoregressive Generation via Score
  Maximization](https://arxiv.org/abs/2505.07812) derives codebook-free
  continuous autoregression from strictly proper scoring rules and motivates
  retaining an energy score rather than converting glyphs to categorical IDs.
- [Autoregressive Image Generation without Vector
  Quantization](https://arxiv.org/abs/2406.11838) shows that a conditional
  continuous density head can replace vector quantization.
- [What Regularized Auto-Encoders Learn from the Data Generating
  Distribution](https://arxiv.org/abs/1211.4246) relates denoising
  reconstruction to the local data score. It motivates measuring whether a
  generated latent is stable after decode and reread instead of trusting an
  unconstrained ambient vector.
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
  supplies a later route for richer glyph-form distributions. V47 first tests
  the cheaper sample generator already compatible with the single-GPU budget.

These papers support components, not the project's language claim. V47 must
still pass image-history controls and autonomous raster gates.

## Selected Visual Coordinate

Let `x` be an ink-positive raster in `[0,1]^(1x32x32)`. V34 was trained with a
white-positive convention, so its frozen EMA encoder `E` receives `1-x`.
Because its output uses non-affine layer normalization, V47 defines

```text
e(x) = E(1 - x)
c(x) = e(x) / ||e(x)||_2                 in S^767.
```

For a unit field `c`, the frozen V34 decoder `D` receives the original latent
radius:

```text
white_logits(c) = D(sqrt(768) * c)
ink_probability(c) = sigmoid(-white_logits(c)).
```

This is a learned continuous visual manifold, not a codebook. `E` and `D`
contain no row per character, accept arbitrary rasters and continuous vectors,
and are frozen during V47.

The visible reread map is

```text
R(c) = c(1[ink_probability(c) >= 0.5]).
```

The recurrent boundary uses `R(c)`, never the unrendered proposal.

## Causal Language Core

For an ordered raster history, the reader computes

```text
h_t = T(c(x_1), ..., c(x_t))
a_t = normalize(H(h_t)).
```

`T` retains V42's eight causal blocks, width 384, six heads, and 64-cell
context. Its input/output field width becomes 768. It is initialized from
scratch; neither V42 nor V46 language weights are loaded.

Natural supervision retains exact-raster multi-positive contrast, anchor
cosine, decoded-pixel, sample energy, and best-sample direction losses. A soft
cycle term decodes each sampled state, rereads the differentiable white
probability through the frozen encoder, and penalizes angular disagreement.
Inference still thresholds before feedback.

## Binding Without Pair Memorization

V43 repeatedly sampled 5,000 training pairs. Its post-result audit reached
`0.9922` on seen pairs but only `0.5830` on unseen training pairs. V47 instead
uses exactly 80,000 unique four-glyph-suffix pairs, once each:

```text
split: train
seed: 20264702
count: 80,000
unique suffixes: 80,000
sequence SHA-256:
2f573c4c79deb9e2bf97c2b0af588a438c7a43280bd632c05c0ae477ec6918eb
```

Each pair has identical final four raster cells, different earlier histories,
and different next raster images. Candidate columns are independently
permuted. The learned module receives only the two image histories, two image
candidates, and a numeric column assignment. This objective cannot be solved
from the shared suffix or a fixed column.

## Autonomous Writer And Closed Loop

The stochastic head produces unit fields `g_r(h_t,a_t,epsilon_r)`. It has no
candidate glyph bank. Each field is decoded to pixels and reread. Selection is
based on the reread field's cosine with `a_t`, so an invisible off-manifold
state cannot win while emitting unrelated pixels:

```text
r_r = R(g_r)
r*  = argmax_r a_t^T r_r
output = binary_decode(g_r*)
next input = output pixels.
```

The evaluator may later identify the raster against a 1,024-image bank, but
that bank is absent while generating and selecting the output.

## Why This Is A Better Test

V47 changes the interface in ways tied directly to measured failures:

- fixed radius removes V46's collapsed scale coordinate;
- the nonlinear V34 retina supplies a glyph-aware manifold with measured
  held-font identity retention;
- 80,000 nonrepeating counterfactuals target long-history binding rather than
  pair memorization;
- visible reread selection and cycle regularization target the latent-to-pixel
  mismatch; and
- the same small V42 causal core keeps the test within one RTX 4090.

It does not yet add instructions, page segmentation, calligraphy control, or a
local LLM teacher. Those additions would hide whether the basic image-language
mechanism works.

## Scale-Up Path If V47 Qualifies

1. Add multiple modern fonts at input while keeping a canonical output motor.
2. Learn a separate image-conditioned form/style state, initialized from V41,
   without changing content identity.
3. Pair Hanziyuan oracle, bronze, seal, and later forms with modern visual
   content states; keep raw unencoded forms as raster outputs.
4. Render prompt/answer curricula prepared offline by the local 4B/8B model.
   The teacher may prepare data but remains absent from the independent ILM.
5. Replace fixed glyph cells with overlapping page patches and learned visual
   fixation only after canonical closed-loop language is stable.

## Claim Boundary

A passing V47 would establish only that a compact model can learn improved
canonical Chinese next-glyph language from raster history and autonomously
emit a stable raster through a frozen continuous visual manifold. It would not
establish instruction following, etymology answers, arbitrary historical-form
generation, human-equivalent reading, Qwen parity, or superior scaling.

