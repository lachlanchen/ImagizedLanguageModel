# Scaled Retinal Glyph Language V46: Frozen Development Protocol

Status: preregistered before V46 implementation, smoke training, or production
training.

Date frozen: 2026-08-15

## Question And Allowed Claim

Can the exact V42 causal image-stream architecture, trained from initialization
in the qualified V45 retinal geometry, improve held-out Chinese continuation,
counterfactual binding, and bank-free next-raster generation without token IDs,
Unicode IDs, OCR, a character classifier, a codebook, or a deployed candidate
bank?

V46 changes the visual coordinate and the output normalization required by that
coordinate. It does not test historical writing, semantics, question answering,
etymology, page reading, cross-font generalization, scaling laws, or parity with
a text LLM.

## Immutable Inputs

- Corpus:
  `data/visual_grammar/chinese_wikisource_public_domain.jsonl`.
- Corpus SHA-256:
  `76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03`.
- Record partition, original-plus-offline-simplified views, and canonical Noto
  Sans CJK rendering are exactly V42's.
- Context and target are the same length-64 shifted raster streams as V42.
- Student batches contain only floating tensors shaped
  `B x 64 x 1 x 32 x 32`.
- V45 checkpoint:
  `artifacts/noise_limited_retinal_field_v45_20260814/field.pt`.
- Required V45 checkpoint SHA-256:
  `0e5947d85a8baeff99d92996ee8434d3aceab39e64042c9b6ec1a142aa174534`.
- Required V45 field-state SHA-256:
  `08b57734ac3ded0c1438cc4bf963d92357ce1f1d31ae49ee6548c56c19db019d`.
- Required V45 fit-sequence SHA-256:
  `eb0a4ede44a062eb4107fa857c0cda488734b9ddf5ad56e30a5f375516e61fd8`.

Strings and labels may exist in host-side loaders and evaluators. They may not
enter the student call, student state, field transform, generator, or deployed
inference path.

## Fixed Scaled Retinal Field

For raster `x`, V46 uses V45's fixed FP64-fit buffers and FP32 runtime transform:

```text
d(x) = DCT2(2 * 1[x >= 0.5] - 1)
z(x) = A(d(x) - mu)
s    = 19.622622215774165
v(x) = z(x) / s.
```

The scale is the FP64 training-count-weighted RMS V45 radius over the frozen
8,000-form fit sequence:

```text
s = sqrt(sum_i w_i ||z(x_i)||^2).
```

It is a persistent scalar buffer and is never learned. The inverse is

```text
d_hat = mu + A^-1(s v_hat)
x_hat = 1[IDCT2(d_hat) >= 0].
```

Direction is computed only when a cosine is required:

```text
u(v) = v / max(||v||, 1e-8).
```

The model must preserve the full unnormalized `v` at input and output. No
1,025th radial coordinate, learned scale, clamp, quantizer, or lookup is allowed.

## Fixed Causal Model

V46 retains V42's learned-layer shapes:

```text
field width             1024
model width              384
causal layers              8
attention heads             6
MLP ratio                   3
maximum context             64 raster cells
noise width                128
generator layers             4
generator samples            4
```

The trainable parameter count must equal V42 exactly: `24,346,497`.
The V45 transform and scalar scale are fixed buffers with zero trainable
parameters.

All linear layers use V42's zero biases and normal weight initialization with
standard deviation `0.02`, except:

```text
final anchor projection weight std      0.008
final generator projection weight std   0.0005
```

The causal reader emits hidden state `h_t` and an unnormalized full-field anchor
`a_t`. The stochastic generator emits

```text
g_r = a_t + residual(h_t, a_t, epsilon_r)
```

without final normalization. Autonomous sampling selects the generated field
with minimum squared Euclidean distance to `a_t`, applies the exact inverse, and
rereads the resulting raster at the next visual time step.

## Fixed Loss

For target field `v`, the directional compatibility is

```text
c(a, v) = alpha * u(a)^T u(v),
```

where `alpha` is V42's learned bounded contrastive scale. Positives are exact
binary-raster matches present in the current minibatch; no persistent bank or
identity label is used.

The losses and weights are:

```text
L_nce    = dynamic multi-positive directional contrastive loss
L_anchor = mean(1 - u(a)^T u(v))
L_pixel  = binary cross entropy + 0.5 * soft Dice after exact inverse
L_energy = 2 E_r ||g_r-v||_2 - E_(r != s) ||g_r-g_s||_2
L_sample = mean(1 - max_r u(g_r)^T u(v))

L = 1.00 L_nce + 0.25 L_anchor + 0.20 L_pixel
  + 0.50 L_energy + 0.25 L_sample.
```

No explicit radius loss is added. `L_energy` supervises the complete field and
`L_pixel` supervises its exact inverse, so adding a separately weighted radial
objective would change the controlled intervention.

## Fixed Optimization

V46 copies V42 exactly:

```text
updates                         10,000
batch size                           8
gradient accumulation                2
effective examples/update           16
optimizer                        AdamW
betas                         (0.9, 0.95)
learning rate                    3e-4
warmup updates                     500
minimum LR ratio                  0.10
weight decay                      0.05
gradient clip                      1.0
maximum contrastive positions       512
maximum energy positions             128
energy samples                         4
precision                           BF16
model seed                    20264200
dataset seed                  20264201
```

The data order is therefore matched to V42. V46 is trained from initialization;
no V42 or V44 learned parameter may be loaded.

## Fixed Development Audit

The audit copies V42 exactly:

- seed `20264220`;
- 2,048 deterministic development windows;
- 1,024 canonical evaluator-only candidate rasters;
- full, suffix-4, last-only, suffix-preserving shuffled, and blank contexts;
- training-only image-unigram and symbolic-bigram controls;
- 512 same-suffix, different-target counterfactual pairs;
- 256 bank-free generated examples with four samples; and
- exact raster rereading in autoregressive inference.

Candidate ranking uses directional cosine. Generated samples are selected before
the evaluator bank is consulted. In addition to V42 metrics, report anchor and
selected-field radius MAE, relative radius MAE, finite rate, exact field
round-trip error, checkpoint/field/protocol digests, elapsed time, VRAM, and
parameter equality.

The comparison baseline is immutable:

```text
V42 checkpoint SHA-256
a5ce2ff20d0fc6d336f489b1e2d29bb96a1d0666399c2e2b95e7ac33a9abe870

V42 development report SHA-256
b376c3ae42c147c6e6ae9ed81f0294240bfb3c2da527b52312040e63a36584f7

V42 full top-1                  0.19970703125
V42 full target log probability -5.255309844389558
V42 pair arm accuracy           0.5302734375
V42 generated identity top-1    0.08203125
V42 generated pixel F1          0.37308064103126526
```

## Conjunctive Gates

V46 qualifies only if every condition is true.

The ten V42 gates remain unchanged:

1. full top-1 exceeds image unigram by more than `0.03`;
2. full top-1 exceeds symbolic bigram by more than `0.01`;
3. full target log probability exceeds shuffled by more than `0.05` nat;
4. full top-1 exceeds shuffled by more than `0.015`;
5. counterfactual full-history arm accuracy exceeds `0.60`;
6. bank-free generated identity top-1 exceeds image unigram;
7. generated binary pixel F1 exceeds `0.55`;
8. generated blank rate is below `0.02`;
9. the image-only student boundary is clean; and
10. peak allocated CUDA memory is below `18 GiB`.

Matched improvement gates are added:

11. full top-1 exceeds V42 by more than `0.01`, hence exceeds
    `0.20970703125`;
12. full target log probability exceeds V42 by more than `0.05` nat, hence
    exceeds `-5.205309844389558`;
13. generated identity top-1 exceeds V42 by more than `0.01`, hence exceeds
    `0.09203125`;
14. trainable parameter count equals `24,346,497`, the V45 field digest and
    scale match this protocol, every audited tensor is finite, the frozen
    partition is unopened, and training plus audit finish within 30 minutes on
    one RTX 4090.

Strict comparisons use epsilon `1e-12`. A gate is not rounded before its
decision.

## Stop And Decision Rules

- Smoke runs validate shapes, gradients, causality, exact inversion,
  checkpoint reload, and evaluator plumbing only. They cannot qualify V46.
- Exploratory runs, if any, must be declared before launch and cannot be
  relabelled as production evidence.
- Once the production run begins, no threshold, loss, initialization, seed,
  data order, audit sample, or selection rule may change in response to results.
- One failing gate makes V46 non-qualifying. Report the failure; do not add pair
  supervision, more updates, more parameters, historical data, or a writer
  under this protocol.
- The frozen evaluation partition and V43 writer remain closed regardless of
  the development outcome.

If all gates pass, V46 supports only this claim: a compact causal model trained
from raster streams in the qualified scaled-retinal coordinate improved matched
V42 Chinese next-glyph continuation and generated raster fields without a
deployed symbolic vocabulary. A separate protocol is still required for
historical writing, instruction following, word-origin answers, and broader
language understanding.
