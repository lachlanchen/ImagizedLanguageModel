# Field-Complete Visual Writer V21 Protocol

Date preregistered: 2026-08-12

Status: fixed before V21 implementation or training

## Question

V20 established that a compact image-only writer can be forced to use a local
continuous retinal field. On its final development endpoint, correct-field
dense F1 exceeded shuffled-field by `0.12177` and zero-field by `0.36128`, and a
quadrant occlusion had locality `1.0`. V20 still failed writer selection because
its field supplied only zero-mean within-block detail while global state supplied
coarse occupancy. Overall F1 was `0.63608`, target cosine was `0.81519`, and the
candidate exceeded its equal-parameter control by only `0.01791` dense F1.

V21 tests the next narrow hypothesis:

> Can a shared, topographic decoder write a complete readable Chinese form when
> every output patch, including coarse occupancy and fine detail, must originate
> from its corresponding continuous retinal cell?

V21 remains an isolated visual actuator. The intended global and local states
are read from a different-font semantic image rather than predicted from prior
visual context. It is not autonomous language generation.

## Visual Language Stream Placement

V21 operates on the `D=1`, one-fixation output case of the Visual Language
Stream

\[
X\in[0,1]^{T\times D\times H\times W\times C}.
\]

It tests the visual motor map needed for a later causal stream. It does not
implement page-scale output, geometric depth, 3D character strings, or character
movies. Those extensions remain forbidden until a 2D predicted-state
write--reread loop passes.

## Student Boundary

The candidate may receive:

- the frozen V16 retina's normalized global continuous state
  `z in R^192` from a different-font semantic image;
- the frozen retina's local continuous field
  `F in R^(192 x 4 x 4)` from that image; and
- a separate continuous style image containing a different character.

The learned path may not receive target pixels as a condition, strings, token
IDs, Unicode IDs, OCR, character labels, a finite glyph or visual codebook,
lookup, a candidate classifier, absolute character identity, or an external
language model. Target ink enters losses only. Offline identifiers and candidate
images are evaluator-only.

The exact frozen PVF checkpoint remains:

```text
artifacts/predictive_visual_field_v16_memory_pilot/checkpoint_step_0002200.pt
SHA-256 90791001203640f0de66316cf2e30b3e2c588480fef0e3d9d4f6283ba043ecbe
```

No V17--V20 writer weight may enter either V21 arm. All trainable parameters
start from the fixed seeded initialization.

## Field-Complete Architecture

Let image-derived style be `s in R^64`. A global conditioner produces one
continuous context shared over every retinal location,

\[
m=M([z,s])\in\mathbb R^{128}.
\]

`m` may only apply channelwise scale and shift to a shared pointwise cell
decoder. It may not produce a spatial seed, coordinate map, position embedding,
attention query per location, transposed-convolution seed, or spatial bias.
There is no learned or fixed absolute position input. Consequently, global state
alone cannot assign different content to different retinal cells.

The candidate source is the local retinal field:

\[
u_{ij}=P(F_{ij}),\qquad i,j\in\{1,\ldots,4\}.
\]

Three shared pointwise residual blocks transform every `u_ij` under the same
global context `m`. No block exchanges information between cells. Each final
cell emits one coarse coefficient and 63 detail coefficients for its
corresponding `8x8` output patch.

### Exact Walsh--Hadamard patch basis

Let `H_64` be the Sylvester Walsh--Hadamard matrix and define

\[
B_0=\frac{1}{8}H_{64}[:,1:64]\in\mathbb R^{64\times63}.
\]

The omitted first column is the constant DC mode. Every retained column has
exactly 32 entries `+1/8` and 32 entries `-1/8`; these values and their zero
sums are exactly representable in binary floating point. For cell `(i,j)`, the
decoder emits coarse scalar `c_ij` and coefficients `a_ij in R^63`:

\[
\ell_{ij}=c_{ij}\mathbf 1_{64}+B_0a_{ij}.
\]

The 16 patches are placed without overlap into a `32x32` logit field and passed
through a sigmoid. Both coarse occupancy and zero-DC detail originate from the
same local retinal cell. The basis is a fixed buffer, not a codebook: it carries
no glyph identity and is shared across every form and location.

The trace must expose `coarse_cell_logits`, `coarse_logits`,
`detail_coefficients`, `detail_logits`, `combined_logits`, `local_source`, and
`global_context`.

## Exact-Parameter Tiled-Global Control

The control replaces every candidate source cell with the same normalized
global state:

```text
candidate source[i,j] = F[:, :, i, j]
control source[i,j]   = z, repeated at all 16 cells
```

Every module, tensor shape, operation, loss, initialization seed, optimizer,
batch, schedule, and update count remains identical. The fixed configuration is:

- visual/spatial channels: `192`;
- style dimension/base channels: `64 / 32`;
- pointwise hidden/context channels: `128 / 128`;
- pointwise residual blocks: `3`;
- dropout: `0.05`; and
- no positional parameters.

The expected trainable parameter count is exactly `582,336` per arm. Unit tests
must verify equality and this absolute count before an evidence run.

With identical tiled source and spatially uniform modulation, the control must
emit the same `8x8` patch at all 16 locations. It is an information-ablation
control, not an admissible final writer. It is selected by best development
quality without a minimum quality gate so a failed control cannot prevent a
valid paired comparison.

## Structural Tests

Before any evidence run, unit and integration tests must require:

1. exact input, coefficient, patch, and output shapes;
2. exactly `582,336` trainable parameters in each arm;
3. exact parameter-shape equality between arms;
4. no positional parameter or spatial global projection;
5. Walsh--Hadamard basis column sums equal zero exactly in FP32;
6. basis Gram matrix within `1e-6` of identity;
7. detail block-mean magnitude below `5e-6` in FP32;
8. combined logits equal coarse plus detail within `1e-6`;
9. zero-source cell-to-cell patch variation below `1e-6`;
10. a one-cell field intervention changes only its matching `8x8` output patch;
11. a quadrant intervention changes only its matching `16x16` quadrant;
12. candidate output and field gradients change with `F`;
13. global and style inputs have no spatially varying path independent of the
    local source;
14. the frozen retina receives no gradient; and
15. checkpoint serialization preserves every output and receipt field.

## Causal Interventions

For fixed target and style, evaluation computes:

1. correct `z`, correct `F`;
2. correct `z`, batch-shuffled `F`;
3. batch-shuffled `z`, correct `F`;
4. both `z` and `F` batch-shuffled together;
5. correct `z`, zero `F`; and
6. correct `z`, with each of four non-overlapping `2x2` field quadrants zeroed.

Quadrant locality is computed exactly as in V20. Because the V21 decoder is
pointwise across cells, a correct implementation should have locality `1.0` up
to numerical equality. The fixed gate is therefore stricter than V20.

## Image-Only Complexity Strata

V21 reuses the V19/V20 target-geometry score without recalibration:

\[
C(x)=\operatorname{mean}(b)
+0.5\left[\operatorname{mean}|\Delta_x b|
+\operatorname{mean}|\Delta_y b|\right]
+0.1\operatorname{mean}
\left[\operatorname{AvgPool}_{4\times4}(b)>0.05\right],
\]

where `b = 1[x >= 0.5]`.

- simple: `C < 0.24`, weight `1.00`;
- medium: `0.24 <= C < 0.35`, weight `1.25`; and
- dense: `C >= 0.35`, weight `2.00`.

No identity or text label enters this score.

## Training Objective

Both arms use

\[
\begin{aligned}
\mathcal L_{21}={}&
\mathcal L_{\mathrm{WBCE}}+
\mathcal L_{\mathrm{Dice}}+0.5\mathcal L_1+
0.25\mathcal L_{\mathrm{edge}}\\
&+0.10\mathcal L_{\mathrm{id}}+0.05\mathcal L_{\mathrm{NCE}}
+0.25\mathcal L_{\mathrm{field\ margin}}
+0.25\mathcal L_{\mathrm{zero\ margin}}
+0.50\mathcal L_{\mathrm{coarse}}.
\end{aligned}
\]

`L_coarse` is binary cross entropy between each field-derived coarse coefficient
and the exact `4x4` average-pooled target. Candidate margins require correct
mean pixel error to beat shuffled-field and zero-field errors by `0.05`. These
margin losses are zero for the tiled-global control. Complexity weights apply
to all per-example topology terms.

Fixed optimizer and run settings:

- AdamW, learning rate `3e-4`, betas `(0.9, 0.95)`, weight decay `0.03`;
- 100-step warmup and cosine decay to `0.1` of initial rate;
- gradient norm clip `1.0`;
- BF16 autocast, batch size `32`;
- `1,600` updates per arm;
- validation every `200` updates on `512` generated development candidates;
- sequence length `24`, four sampled positions per sequence;
- training seed `20260821`;
- one RTX 4090 per arm; and
- no hyperparameter or threshold change after the first non-smoke candidate
  run.

Smoke mode may run at most 20 updates and is permanently non-evidentiary. A
smoke checkpoint cannot resume into an evidence run.

## Partition

Partition salt: `field-complete-writer-v21`.

Use the first 64 bits of
`sha256(salt + NUL + identifier)`, holdout fraction `0.06`, and equal
development/frozen shares. The receipt records exact counts and the SHA-256 of
sorted frozen identifiers. Neither arm, model selection, paired audit, nor
blinded review may instantiate a frozen image.

The V19 and V20 frozen partitions remain sealed and are never reused.

## Candidate Selection

Rank eligible candidate checkpoints by overall correct pixel F1, then dense F1,
then earlier step. Eligibility requires all of:

1. overall correct pixel F1 `> 0.66`;
2. simple correct pixel F1 `> 0.58`;
3. medium correct pixel F1 `> 0.60`;
4. dense correct pixel F1 `> 0.70`;
5. dense correct-minus-shuffled-field F1 `> 0.15`;
6. dense correct-minus-zero-field F1 `> 0.20`;
7. correct identity top-1 `> 0.74` and above both-shuffled top-1;
8. correct target cosine `> 0.82` and above both-shuffled cosine;
9. correct-versus-shuffled-field pixel L1 `> 0.12`;
10. correct-versus-zero-field pixel L1 `> 0.15`;
11. mean quadrant-occlusion pixel change `> 0.03`;
12. mean quadrant-occlusion locality `> 0.95`;
13. style-copy cosine `< 0.30`;
14. semantic-reference versus target pixel L1 `> 0.05`;
15. fixed-basis DC leakage equals `0.0` in FP32;
16. zero-source cell variation `< 1e-6`;
17. detail block-mean magnitude `< 5e-6`;
18. decomposition error `< 1e-6`; and
19. no frozen image is instantiated.

These gates are development-only. Passing them selects a candidate but does not
authorize a frozen query.

## Control Selection And Paired Gate

The tiled-global control checkpoint is ranked by overall development F1, then
dense F1, then earlier step. It has no quality minimum. It must still pass the
structural basis, repeated-cell, decomposition, equal-parameter, image-only, and
sealed-frozen checks.

After candidate and control selection, evaluate both once on the same fresh
development renderings. The paired gate requires:

1. candidate overall F1 exceeds control overall F1 by `> 0.20`;
2. candidate dense F1 exceeds control dense F1 by `> 0.20`;
3. candidate retains every arm-specific gate on the fresh audit;
4. control retains every structural/receipt gate; and
5. trainable parameter counts are exactly equal.

No endpoint comparison may substitute for selected checkpoints.

## Blinded Readability Gate

Only after candidate and paired automatic gates pass, generate one fixed
48-example development audit: 16 simple, 16 medium, and 16 dense outputs. Show
each output beside eight randomly ordered different-font visual candidates,
without text labels or target rows. Score visual duplicate groups using the
frozen retina's precomputed image-only duplicate mask.

Required recognition is:

- simple: at least `14/16`;
- medium: at least `12/16`;
- dense: at least `10/16`; and
- total: at least `38/48`.

Answers and keys are stored separately and scored by code. Examples may not be
replaced after review.

## Frozen Policy

V21 has a new salted frozen partition. Frozen evaluation is forbidden until the
candidate selection gate, fresh paired gate, and numeric blinded-readability
gate all pass. If any gate fails, V21 is documented as development-only and its
frozen images remain uninstantiated.

If all gates pass, one immutable frozen writer evaluation is permitted. It must
be run by a separate script that verifies checkpoint hashes, protocol payload,
student boundary, selected-step receipts, and prior gate receipts before
constructing the frozen dataset.

## Claims

After candidate, paired, blinded, and frozen success, allowed claims are limited
to:

- a compact continuous retinal field can causally determine complete readable
  Chinese topology under the named V21 intervention;
- a shared local visual route outperforms an exact-parameter global-only source
  on the named partitions; and
- measured one-GPU parameters, memory, throughput, and image quality.

Even after success, forbidden claims include autonomous language generation,
general understanding, historical factuality, page-scale generation, 3D or
movie generation, Qwen parity, or efficiency superiority. Those require
predicted rather than supplied fields, a readable closed write--reread stream,
and later task benchmarks.
