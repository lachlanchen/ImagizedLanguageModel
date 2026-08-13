# Conditional Visual Field Flow V31 Protocol

Date frozen: 2026-08-13

Status: preregistered before V31 implementation, smoke testing, optimization,
or development measurement

## Decision Question

Can a compact image-only causal model learn a coherent conditional flow over
the full next-image retinal field, bind earlier visual history to the correct
arbitrary next glyph, and outperform an identically initialized position-blind
global control?

V31 is a two-arm development experiment: `spatial-field` and
`global-control`. Both arms must be implemented, trained, and audited. Smoke
runs test plumbing only. They cannot revise this protocol, select a checkpoint,
or provide language evidence. No frozen image may be instantiated and no pixel
writer may be trained unless every gate explicitly required below passes.

## Fixed Prior Evidence And Initialization

Both arms initialize from the completed V30 global-control final checkpoint:

`artifacts/spatial_visual_next_field_v30_global_control_evidence/checkpoint_final.pt`

Required SHA-256:

`66378d4b972702490f6819d87d95c2576546e15e6fc74d10542307aaf4483411`

The source provides the frozen V16 retina, frozen semantic adapters, and the
stronger of the two trained V30 causal visual backbones. Its measured natural
top-1 was `0.02490234375`; exact-suffix pair assignment was
`0.505859375`. The source is rejected evidence, not a selected language model.

Both V31 arms load exactly these source modules:

- `retina`;
- `semantic_adapter`;
- `target_semantic_adapter`;
- `context_input`;
- all eight `context_blocks`; and
- `context_norm`.

The V30 `field_decoder` and `logit_scale` are discarded. The retained retina
and both semantic adapters remain frozen. Context modules remain trainable.
The source retina receipt must remain
`90791001203640f0de66316cf2e30b3e2c588480fef0e3d9d4f6283ba043ecbe`.

## Fixed Data

- manifest: `data/visual_grammar/chinese_wikisource_public_domain.jsonl`;
- required manifest SHA-256:
  `76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03`;
- identifier and font partitions: unchanged from V25-V30;
- train script views: original and OpenCC simplified when distinct;
- development script views: original and simplified;
- context images: 64;
- target images: 1;
- authoritative natural shape: `[65,1,32,32]` per font view;
- exact suffix: final 4 context images;
- rendering: the existing 32-cell renderer, two training fonts, font size
  24-28, at most one-pixel translation, and existing bounded
  intensity/blur/noise augmentation;
- canonical equality witness: unaugmented Noto Sans CJK Regular pixels;
- training suffix pairs: the same 32,768 deterministic V30 specifications;
- route arms: exactly the same windows, pairs, rendered tensors, augmentations,
  cross-font directions, candidate orders, and update count; and
- frozen identifiers and fonts: sealed.

Every training pair has two different record identifiers, different targets,
the same final four writing forms, and bitwise-equal rendered suffix pixels.
Candidate order is independently randomized per cross-font direction.

### Host-side candidate sets

Natural training uses only target images already present in the fixed V30
1,024-image training bank, but each microbatch uses its own opposite-font
target images as the candidate set. The persistent 1,024-image bank is not
passed through the flow during training.

Development uses the unchanged evaluator-only 1,024-image bank and the same
2,048 natural windows and 512 suffix pairs as V30. Candidate images and
canonical equality labels remain host-side. No candidate image, form, or label
is stored in a final checkpoint or required for autonomous generation.

## Fixed Seeds

| Purpose | Seed |
|---|---:|
| common model initialization | `20261110` |
| natural training dataset | `20261111` |
| suffix-pair training dataset | `20261112` |
| training flow probes | `20261113` |
| deterministic development audit | `20261114` |

Each route starts in a fresh process from the same source and common model
seed. Before update one, all parameter keys, values, shapes, dtypes, and model
state SHA-256 must be byte-identical. Route identity affects gradients only
through the fixed target-field construction.

## Fixed Student Boundary

Allowed learned-path tensors:

- context images `[B,T,1,32,32]`, `1 <= T <= 64`;
- arbitrary target or candidate images ending in `[1,32,32]`;
- frozen continuous visual fields computed from those images;
- floating times, coherent Gaussian base fields, noisy fields, velocity
  fields, and generated fields;
- temporary positional candidate columns used by losses; and
- renderer augmentation values.

Forbidden learned-path values:

- strings, bytes, text transcripts, or OCR output;
- token, Unicode, character, vocabulary, glyph, or record IDs;
- tokenizer or discrete visual-codebook indices;
- vocabulary embeddings or vocabulary logits;
- glyph lookup or a deployed candidate bank;
- symbolic n-gram values;
- teacher logits or hidden states; and
- external language-model calls.

Host data and evaluation code may use source strings to select windows and
render images, exact canonical pixels to locate columns, and symbolic controls
for comparison. Those values never enter a model method or checkpoint.

## Fixed Representation And Routes

The input is an ordered image stream

\[
X\in[0,1]^{B\times T\times1\times32\times32}.
\]

For arbitrary candidate image `y`, fixed perception returns raw vector `r(y)`,
semantic vector `z(y)`, and channel-normalized local retinal cells
`R_p(y)`, `p=1,...,16`.

The route target is

\[
Y_p(y)=
\begin{cases}
R_p(y), & \text{spatial-field},\\
z(y), & \text{global-control}.
\end{cases}
\]

Both targets have shape `[16,192]`; every row has unit norm. No route-specific
learned parameter exists.

## Fixed Coherent Base And Probability Path

Draw `e ~ N(0,I_192)`, normalize it to unit length, and tile it over all 16
retinal positions:

\[
E_p=e,\qquad p=1,...,16.
\]

The same coherent base law is used for both routes, training, path scoring, and
generation. Independent per-patch noise is forbidden in V31.

For training target `Y`, sample `t ~ Uniform(0.001,0.999)` and define

\[
Z_t=(1-t)E+tY,\qquad U^*=Y-E.
\]

The conditional flow-matching loss is

\[
L_{CFM}(h,Y)=\frac1{16}\sum_p
\lVert v_\theta(h,Z_t,t)_p-U^*_p\rVert_2^2.
\]

The endpoint estimate used only for a training diagnostic is

\[
\tilde Y=\operatorname{normalize}_{channel}
[Z_t+(1-t)v_\theta(h,Z_t,t)].
\]

## Fixed Architecture

Architecture name: `conditional-visual-field-flow-v31`.

| Component | Fixed value |
|---|---:|
| cell size | 32 |
| maximum context | 64 images |
| exact suffix | 4 images |
| raw retinal width | 192 |
| semantic width | 192 |
| causal width | 384 |
| causal blocks | 8 |
| causal heads | 6 |
| causal MLP ratio | 3.0 |
| causal dropout | 0.05 |
| field shape | `4 x 4 x 192` |
| coherent base width | 192 |
| velocity hidden channels | 192 |
| velocity residual blocks | 4 |
| spatial kernel | depthwise `3 x 3` |
| channel MLP ratio | 2.0 |
| velocity dropout | 0.05 |
| sinusoidal time width | 128 |
| initial path-score temperature | 0.25 |
| maximum inverse temperature | 100 |
| total-parameter cap | 20 million |
| trainable-parameter cap | 18.5 million |

The retained V30 causal reader computes

\[
h=C_\theta([r(x_1);z(x_1)],\ldots,[r(x_T);z(x_T)])_T
\in\mathbb R^{384}.
\]

The velocity decoder:

1. projects each noisy 192-vector to 192 hidden channels;
2. adds one learned `4x4x192` retinal-position tensor;
3. combines layer-normalized `h` with a two-layer SiLU projection of a
   128-dimensional sinusoidal embedding of `t`;
4. applies four residual blocks, each consisting of channel layer norm,
   depthwise `3x3` mixing, condition-derived FiLM scale and shift, a
   `192 -> 384 -> 192` pointwise MLP, and dropout; and
5. projects to an unconstrained `4x4x192` velocity.

FiLM scale uses `1 + 0.1*tanh(gamma)`. All new linear and convolutional weights
use normal initialization with standard deviation `0.02`, zero biases, and
residual output scaling `1/sqrt(8)`. The position tensor uses standard
deviation `0.02`. The learned inverse path-score temperature starts at 4 and
is clamped to 100.

Self-conditioning, classifier-free guidance, learned schedules, reflow,
distillation, attention inside the velocity decoder, and route-specific
parameters are forbidden.

## Fixed Conditional Path Score

Training uses two coherent probe vectors per microstep. Development uses eight
fixed coherent probes. A probe vector is shared across every candidate column
and context row in the compared score matrix.

Training probe times are `(0.10, 0.35)`. Development probe times are

`(0.03, 0.07, 0.12, 0.20, 0.30, 0.42, 0.55, 0.70)`.

For encoded context `h`, candidate field `Y`, probe base `E_m`, and time `t_m`,

\[
D_m(h,Y)=\frac1{16}\sum_p
\left\lVert
v_\theta(h,(1-t_m)E_m+t_mY,t_m)_p-(Y_p-E_{m,p})
\right\rVert_2^2,
\]

\[
s_{path}(h,Y)=-\tau\frac1M\sum_m D_m(h,Y).
\]

This score is a fixed denoising/flow surrogate, not a claimed exact log
likelihood. Shared probes make complete candidate-column permutation exactly
equivariant. Candidate scoring is chunked at 32 candidates in development.

## Fixed Context Corruption And Spatial Intervention

The suffix-preserving shuffle independently permutes context positions `0..59`
and preserves positions `60..63`; identity permutations become one-step
rotations. Candidate pixels and path probes remain unchanged.

The spatial candidate intervention reverses row-major retinal positions:

\[
\pi=(15,14,\ldots,1,0).
\]

It permutes candidate target rows and target-velocity rows while leaving the
coherent tiled base unchanged. The global target is tiled, so this operation
must be exactly invariant in the global-control arm. It is evaluator-only.

## Fixed Training Losses

Every term averages first-font-to-second-font and second-font-to-first-font
directions.

### Natural microbatch

For `B=4` contexts and their opposite-font target images:

- compute correct-target `L_natural-CFM` with one fresh uniform time/base;
- use all four target images as arbitrary in-batch candidate columns;
- compute full and suffix-preserving shuffled path-score matrices with the
  same two probes;
- apply row-wise cross entropy `L_natural-CE` to exact positional targets; and
- apply

\[
L_{natural-order}=\operatorname{softplus}
[0.10-(s_{full,target}-s_{shuffle,target})].
\]

### Exact-suffix pair microbatch

For `B=4` two-context/two-candidate pairs:

- gather the correct route field for each context and compute `L_pair-CFM`;
- compute full and shuffled `2x2` path-score matrices with identical probes;
- apply the existing symmetric row/column assignment loss `L_pair-assign`;
- apply correct-minus-other margin loss

\[
L_{pair-positive}=\operatorname{softplus}(0.10-m_{full});
\]

- and apply order gain loss

\[
L_{pair-order}=\operatorname{softplus}
[0.10-(m_{full}-m_{shuffle})].
\]

Suffix-only score rows are measured and must be exactly equal; suffix scores
are not subtracted from full scores.

### Total loss

\[
L=L_{natural-CFM}+L_{natural-CE}+0.5L_{natural-order}
+L_{pair-CFM}+4L_{pair-assign}+L_{pair-positive}+L_{pair-order}.
\]

No ODE solve occurs during training. No symbolic baseline, form identity,
candidate ID, or teacher value enters a loss.

## Fixed Optimization

Each arm runs independently:

| Setting | Fixed value |
|---|---:|
| optimizer | AdamW |
| updates | 10,000 |
| natural batch | 4 per microstep |
| suffix-pair batch | 4 per microstep |
| gradient accumulation | 2 |
| effective natural rows | 16 per update across views |
| effective pair rows | 32 per update across views and pair arms |
| velocity learning rate | `3e-4` |
| context learning rate | `6e-5` |
| warmup | 500 updates |
| schedule | cosine to 10% of base rate |
| weight decay | 0.05 |
| global gradient clip | 1.0 |
| precision | BF16 autocast |
| checkpoint interval | 1,000 updates |
| checkpoint selection | none; final update only |

The arms may run sequentially on one RTX 4090. They share no optimizer state,
gradients, or learned tensors. No fixed setting may change in an evidence run.

## Fixed Autonomous Generation

After one context encoding, draw eight coherent base vectors from the audit
seed and integrate from `t=0` to `t=1` with eight uniform Heun steps. Each step
uses the same trained velocity field; no candidate image is present. Final
rows are channel-normalized.

For generated fields `G_k` and candidate field `Y`, define

\[
s_{sample}(h,Y)=\log\frac1{8}\sum_{k=1}^{8}
\exp\left(16\cdot\frac1{16}\sum_pG_{k,p}^{\top}Y_p\right).
\]

The evaluator reports sample-score natural and pair assignment, best-of-eight
target rank, mean pairwise sample cosine distance, full-versus-shuffled
same-noise displacement, and nearest candidate images. The 1,024-image bank is
used only after generation and never enters the ODE.

## Fixed Development Audit

One evaluator compares both final checkpoints on:

- 2,048 deterministic natural development windows;
- 512 deterministic exact-suffix development pairs;
- the unchanged two-font 1,024-image evaluator bank;
- path scores under full, suffix-4, shuffled, and spatially permuted inputs;
- autonomous sample scores under full and shuffled contexts with identical
  base draws;
- candidate-column permutations;
- aligned spatial and global target visibility;
- image unigram and symbolic bigram/trigram controls built outside the model;
- initialized-state, source, protocol, data, pair, font, and window receipts;
- parameter, checkpoint, finiteness, wall-time, and peak-VRAM receipts; and
- proof that no frozen image was instantiated.

Both arms use exactly the same windows, image pixels, pair rows, candidates,
probe vectors, sample base draws, and batches.

## Fixed Gates

All strict inequalities remain strict. Tolerance is used only where stated.

### Spatial common mechanism gates

1. every parameter, training metric, path score, velocity, and sample is
   finite;
2. the student boundary and final checkpoint are clean;
3. the context-only sampler emits `[B,K,16,192]` without candidates;
4. no candidate bank, image, form, label, optimizer, or RNG state occurs in
   the final checkpoint;
5. total parameters are `<20,000,000` and trainable parameters are
   `<18,500,000`;
6. peak allocated CUDA memory is `<18 GiB`;
7. aligned spatial target cross-font top-1 is `>=0.95`;
8. pair suffix pixels are bitwise equal;
9. suffix path-score row maximum error is `<1e-6`;
10. candidate-column permutation score error is `<1e-5` and recovered
    accuracy agreement is exactly `1.0`;
11. full path-score pair arm accuracy is `>0.65`;
12. full path-score pair both-correct rate is `>0.40`;
13. full minus shuffled pair arm accuracy is `>0.10`;
14. full minus shuffled pair mean margin is `>0.05`;
15. natural full minus shuffled target log probability is `>0.03` nat;
16. spatial permutation reduces natural target log probability by `>0.05`
    nat;
17. spatial permutation reduces pair arm accuracy by `>0.05`;
18. generated sample mean pairwise cosine distance is in `(1e-4,1.5)`; and
19. same-noise full-versus-shuffled generated-field displacement is `>0.01`.

### Global-control integrity gates

1. all common finiteness, boundary, checkpoint, parameter, and resource gates
   pass;
2. frozen global target cross-font top-1 is `>=0.95`;
3. suffix pixels and suffix score rows are exact;
4. candidate-column permutation equivariance passes; and
5. candidate spatial-permutation maximum path-score error is `<1e-6` and
   sample-score error is `<1e-6`.

The global arm need not pass language gates.

### Matched-arm gates

1. initialized parameter keys, values, shapes, dtypes, and state SHA-256 are
   exactly equal;
2. final total/trainable parameter counts are exactly equal;
3. source, corpus, pair, font, probe, sample, and audit receipts are equal;
4. both arms complete exactly 10,000 finite updates;
5. spatial path-score pair arm accuracy minus global accuracy is `>0.05`;
6. spatial path-score pair both-correct minus global is `>0.05`;
7. spatial natural path-score top-1 minus global is `>0.01`; and
8. spatial natural path target log probability minus global is `>0.05` nat.

### Spatial language and generation gates

1. natural full path-score 1,024-way top-1 is `>=0.15`;
2. full path top-1 minus suffix-4 top-1 is `>0.03`;
3. full path top-1 minus shuffled top-1 is `>0.03`;
4. full path top-1 minus image-unigram top-1 is `>0.03`;
5. full path top-1 minus symbolic-bigram top-1 is `>0.01`;
6. full path target log probability minus symbolic-bigram target log
   probability is `>0.05` nat;
7. exact-suffix path pair arm accuracy is `>0.65` and both-correct is `>0.40`;
8. autonomous sample-score 1,024-way top-1 is `>=0.05`;
9. autonomous sample-score top-1 minus shuffled sample-score top-1 is
   `>0.02`; and
10. autonomous sample-score exact-suffix pair arm accuracy is `>0.60`.

V31 is selected only if every spatial common, global integrity, matched-arm,
and spatial language/generation gate passes. Otherwise it is rejected, the
frozen partition stays sealed, and writer training remains unauthorized.

## Fixed Artifacts

Evidence directories:

- spatial arm:
  `artifacts/conditional_visual_field_flow_v31_spatial_evidence`;
- global control:
  `artifacts/conditional_visual_field_flow_v31_global_control_evidence`; and
- comparison:
  `artifacts/conditional_visual_field_flow_v31_evidence`.

Smoke directories use the same names with `_smoke`.

Each arm writes atomic step checkpoints, a stripped `checkpoint_final.pt`, and
`train.jsonl`. The comparison writes `development_audit.json`,
`comparison_receipt.json`, and a bounded generated-sample contact sheet.

Final checkpoints contain only architecture/route/config, model tensors,
fixed update arguments, source/protocol/corpus/font/pair/probe receipts,
boundary and parameter receipts, training summary, and smoke/exploratory
flags. They contain no optimizer, RNG, candidate, target image, form string,
canonical pixels, or development/frozen example.

## Fixed Run Order

1. Commit this protocol and record its SHA-256.
2. Implement the model, objectives, trainer, evaluator, and tests without
   editing this protocol.
3. Run unit tests and CPU/CUDA smoke runs for both arms.
4. Run deterministic boundary, equivariance, generation, and throughput
   audits; smoke artifacts cannot be promoted.
5. Commit and push the verified implementation.
6. Train the spatial arm on one RTX 4090.
7. Train the global arm from the identical initialization.
8. Run the joint development audit only after both final checkpoints exist.
9. Publish the result and an experiment figure whether it passes or fails.
10. Update the paper with measured values and keep concept figures explicitly
    labeled as designs rather than evidence.
11. Open frozen data only if the joint audit authorizes it. A pixel writer
    still requires a separate protocol.

## Stop And Reject Conditions

Stop and reject an evidence run if a fixed hash, argument, seed, partition,
font, probe, base-noise receipt, state initialization, parameter count, or
update count differs; a forbidden value enters the model/checkpoint; a tensor
or gradient becomes non-finite; memory exceeds the cap; a frozen image is
instantiated; or any threshold changes after a V31 development value is seen.

Interrupted evidence may resume only from an atomic resumable checkpoint with
optimizer and RNG state. That state must be stripped from the final checkpoint.
A clean restart from common initialization is valid. Smoke and partial runs
are never evidence.

## Permitted Claims

On a complete pass, the project may claim only that, on this fixed development
benchmark, a compact image-only causal student learned a coherent continuous
next-field flow; its autonomous samples and arbitrary-image path score bound
earlier visual history to next writing; and aligned retinal topology exceeded
an identically initialized global control.

A pass does not establish a general ILM, human-like reading, instruction
following, etymology knowledge, historical-form generation, page generation,
Qwen-8B parity, or efficiency superiority over token LMs. A failure must be
published as a negative result with the frozen split sealed.
