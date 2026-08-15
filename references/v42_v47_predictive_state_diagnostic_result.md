# V42--V47 Predictive-State Diagnostic Result

Date: 2026-08-15

Status: completed post-result diagnostic; no V42 or V47 gate changed

## Answer

The diagnostic finds a plausible next path, but it also separates two distinct
V47 failures that must not be conflated:

1. V47 has a sharp, position-specific failure at the final position of its
   64-cell context. This is consistent with its only objective that is applied
   exclusively to that absolute position.
2. Removing that cliff would not make V47 competitive with V42. Before the
   cliff, V47 reaches only about 10.5% development top-1, while V42's direct
   visual field reaches about 18%. The frozen V34 codec is therefore retained
   as a strong visual retina/actuator, not as the next causal language
   coordinate.

The smallest supported V48 direction is to return to V42's fixed, invertible
image field and train one compact causal reader to predict several future
visual fields densely at every valid position. It must not use a special
length-64 loss, a discrete vocabulary, Unicode, OCR, a codebook, or a deployed
candidate bank.

This is the best evidenced path currently available in this repository. It is
not yet a successful V48 result.

## Frozen Evidence

The registered diagnostic scored the same two production checkpoints on exact
training- and development-partition reservoirs, each containing 2,048
64-cell contexts and one next-cell target.

| Evidence | SHA-256 |
| --- | --- |
| diagnostic report | `c66c619c41d501e90a4e2b252714cd65b3fe26f9cf47b6dd20730c73afe7d14e` |
| terminal-position report | `a44a69d7c407fe2e6701dcd5f23d48acb2746e324fb5d41206ecf012405bf388` |
| V42 checkpoint | `a5ce2ff20d0fc6d336f489b1e2d29bb96a1d0666399c2e2b95e7ac33a9abe870` |
| V47 checkpoint | `9bec353a278c56aa79f538a22fd0143f3d60b9366a7f782399941811d12663fc` |
| corpus manifest | `76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03` |
| evaluator bank | `a88509b3c1d3093e63dd1ceb77dcd86c7ef282c80927284b93c4ef09cd9456ad` |
| training windows | `42fc41a00a90113dfc1df44f1f435d84db61dfef5fdfc1c4d3f0e6d966958b38` |
| development windows | `e68db6c27b75a4d502a0dac7fe08487f1c5e3850d46ea1e561c7f18a5757e630` |

The main report passed every registered integrity check: strict production
checkpoint loading, common corpus and renderer, complete windows, finite
floating-image inputs, absent gradients, excluded evaluator metadata, and an
unopened frozen partition. The main diagnostic took 74.23 seconds; the focused
terminal intervention took 41.77 seconds on CUDA device 0 with BF16 inference.

## Context-Length Result

### Development top-1

| Visible cells | V42 direct field | V47 codec sphere |
| ---: | ---: | ---: |
| 1 | 0.08105 | 0.05615 |
| 2 | 0.11377 | 0.07373 |
| 4 | 0.14307 | 0.07861 |
| 8 | 0.16309 | 0.09668 |
| 16 | 0.17236 | 0.09961 |
| 32 | 0.17627 | 0.10498 |
| 64 | 0.18018 | 0.02734 |

V42 improves steadily and remains stable. V47 also improves through 32 cells,
but its full 64-cell result collapses. This is not monotonic long-context
degradation and is not explained by a generally incapable reader.

### Training top-1

| Visible cells | V42 direct field | V47 codec sphere |
| ---: | ---: | ---: |
| 1 | 0.06836 | 0.05225 |
| 2 | 0.11914 | 0.07227 |
| 4 | 0.15332 | 0.08838 |
| 8 | 0.18018 | 0.09082 |
| 16 | 0.19385 | 0.09375 |
| 32 | 0.19336 | 0.10156 |
| 64 | 0.19873 | 0.02051 |

The same V47 cliff appears on the training-record partition. At 64 cells the
development-minus-training top-1 gap is `+0.00684`, and the target-log-
probability gap is `-0.00065`. The dominant failure is therefore not a
train-to-development generalization gap.

## Terminal-Position Intervention

The focused intervention reused the exact registered 2,048 development
windows and evaluator bank and varied only the visible suffix length from 56
through 64.

| Visible cells | V42 top-1 | V47 top-1 | V47 most-common prediction |
| ---: | ---: | ---: | ---: |
| 56 | 0.17822 | 0.10742 | 0.03857 |
| 57 | 0.17822 | 0.10693 | 0.03711 |
| 58 | 0.17871 | 0.10742 | 0.03809 |
| 59 | 0.17920 | 0.10596 | 0.03711 |
| 60 | 0.18018 | 0.10498 | 0.03809 |
| 61 | 0.18018 | 0.10693 | 0.04004 |
| 62 | 0.18115 | 0.10840 | 0.03955 |
| 63 | 0.18213 | 0.10498 | 0.03711 |
| 64 | 0.18018 | 0.02734 | 0.32178 |

From 63 to 64 cells, V47 loses 7.76 percentage points of top-1, its distinct
top-1 predictions fall from 380 to 163, and one prediction's frequency jumps
from 3.71% to 32.18%. V42 changes by only -0.20 percentage points and retains
597 distinct predictions.

This scan was selected after the main diagnostic exposed the 64-cell anomaly.
It is explicitly exploratory, changes no gate, and supports localization of
the failure rather than a new model claim.

## Why The Length-64 Objective Is The Leading Cause

V47's ordinary next-image objective supervises every causal position in a
64-cell stream and averages over the flattened positions. Its independent
counterfactual pair loss does something structurally different:

```text
ordinary objective: positions 1 ... 64, averaged
pair objective:     position 64 only, full cross-entropy added every update
```

The pair path always calls `language(context)[..., -1]` on a context whose
shape is fixed to 64 cells. Its unscaled mean cross-entropy is added to the
complete natural objective with coefficient one. Thus the final RoPE position
receives a dedicated objective comparable at the update level to the whole
position-averaged natural objective, while no other absolute position receives
that pair gradient. The pair curriculum itself remained close to chance.

The code-level asymmetry predicted a defect specifically at position 64; the
measured defect occurs specifically between lengths 63 and 64 in both
partitions, while V42's architecturally matched RoPE path has no cliff. This is
strong convergent evidence that the terminal pair objective poisoned the
final-position state.

It is not a mathematical proof of unique causation. A clean ablation would
require retraining V47 without the pair objective. That retraining is not the
next best use of one-GPU time because the pre-cliff V47 state is already much
worse than V42.

## Representation Diagnosis

The 1,024-field geometry rules out simple codebook collapse or one dominant
nearest-neighbour hub. Both fields retrieve themselves perfectly, and both
have low maximum nearest-neighbour hub fractions (`0.00586`). Their geometry
is nevertheless very different.

| Geometry | V42 direct field | V47 codec sphere |
| --- | ---: | ---: |
| centroid norm | 0.71061 | 0.33170 |
| off-diagonal cosine mean | 0.50448 | 0.10915 |
| nearest-neighbour cosine | 0.75438 | 0.56853 |
| self-to-neighbour margin | 0.24562 | 0.43147 |
| centered effective rank | 156.96 | 118.67 |
| PC1 variance fraction | 0.04034 | 0.03529 |

V47's state is more identity-separated, but that does not make it a better
predictive language coordinate. It removes much of the shared visual
similarity available in V42 and has lower centered effective rank. On the
development partition at 64 cells, V42's ordered history beats the shuffled
history by `+0.01514` top-1 and `+0.15834` target log probability. V47 gains
only `+0.00439` top-1 and `+0.00324` target log probability. Its long state is
therefore much less sensitive to useful order.

At 64 cells V42's anchor remains close to its four-cell anchor (cosine
`0.95002`), while improving the true-target rank in 51.56% and worsening it in
32.23% of windows. V47's corrupted terminal anchor has cosine `0.70516` to its
four-cell anchor, improves 37.84%, and worsens 60.16%. At 32 cells, before the
cliff, V47 remains stable but still substantially weaker than V42.

## V48 Decision

The evidence supports the following minimal architecture test:

1. Keep the exact, fixed, invertible V42 raster field as the visual carrier.
2. Keep a small causal visual transformer and image-only recurrent boundary.
3. Predict the next four visual fields from every valid causal position, so
   future structure shapes the hidden state densely rather than through a
   special final-position task.
4. Preserve a strong next-field objective for horizon one and add declared,
   fixed weights for horizons two through four.
5. Generate a short correlated raster-field block through a continuous proper
   scoring objective; decode it directly to pixels and reread only visible
   pixels during autonomous rollout.
6. Keep the 1,024-raster bank strictly evaluator-side.
7. Add an explicit terminal-stability gate comparing context lengths 63 and
   64.

V48 must not repeat V31's one-glyph flow, V35's raw reconstruction-latent
regression, V44's special pair residual, or V47's normalized codec-sphere
language state. It must be trained from scratch, fit one RTX 4090, and be
judged against the frozen V42 checkpoint on the same renderer, corpus, and
development windows.

## Claim Boundary

This result establishes a reproducible failure mechanism and a better
controlled next experiment. It does not establish semantics, instruction
following, etymology answers, arbitrary historical-form generation, human-like
reading, superiority to token LLMs, or parity with Qwen. Those remain scale-up
targets only after a compact image-only model passes continuation, order,
counterfactual, autonomous-raster, and terminal-stability controls.
