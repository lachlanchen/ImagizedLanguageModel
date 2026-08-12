# Retinal Topology Router V20 Protocol

Date preregistered: 2026-08-12

Status: fixed before V20 implementation or training

## Question

V19 exposed a continuous `4x4x192` retinal field but attached it as an optional
residual to a complete global writer. On fresh development data, the correct
field improved dense pixel F1 by only `0.00879` over a shuffled field and
`0.00535` over a zero field. V20 tests a narrower architectural hypothesis:

> Can a compact writer preserve dense Chinese topology when fine spatial detail
> is structurally restricted to the continuous local retinal field?

V20 is an isolated visual actuator experiment. It is not autonomous language
generation because both global and local intended states are read from a
different-font semantic image rather than predicted from prior context.

## Student Boundary

The candidate writer may receive:

- the frozen V16 retina's normalized global continuous state `z` from a
  different-font semantic image;
- the frozen retina's continuous `4x4x192` field `F` from the same image; and
- a continuous style image containing a different character.

It may not receive target pixels as a condition, strings, token IDs, Unicode
IDs, OCR, character labels, a finite glyph or visual codebook, lookup, a
candidate classifier, or an external language model. Target ink enters losses
only. Offline answer keys and candidate images are evaluator-only.

The exact frozen PVF checkpoint is
`artifacts/predictive_visual_field_v16_memory_pilot/checkpoint_step_0002200.pt`,
SHA-256:

```text
90791001203640f0de66316cf2e30b3e2c588480fef0e3d9d4f6283ba043ecbe
```

No V17, V18, or V19 writer weight may enter either V20 arm. Every V20 trainable
parameter starts from the seeded initialization.

## Topology-Necessary Decomposition

Let the frozen retina expose

\[
z\in\mathbb S^{191},\qquad F\in\mathbb R^{192\times4\times4},
\]

and let `s` be a continuous style vector read from the separate style image.
The global branch emits only one coarse logit per retinal cell:

\[
c=C_\theta(z,s)\in\mathbb R^{1\times4\times4}.
\]

`U` repeats each cell over its corresponding `8x8` output block. It has no
learned upsampling or post-upsampling convolution. Therefore `U(c)` is constant
inside each block and cannot represent a thin stroke or edge there.

The field branch uses shared pointwise transformations to decode each retinal
cell into one `8x8` detail patch:

\[
r=H_\phi(F,s)\in\mathbb R^{1\times32\times32}.
\]

`H` may receive style modulation but may not receive `z`. Let `D` be exact
non-overlapping `8x8` average pooling and define

\[
Q(r)=r-U(D(r)).
\]

The final logits are

\[
\ell=U(c)+Q(r),\qquad p=\sigma(\ell).
\]

By construction, `D(Q(r)) = 0` up to floating-point tolerance, and the global
branch has no within-block variation. Fine detail can therefore enter only
through the local retinal field. There is no learned scalar gate that can turn
the local route into a small optional correction.

The implementation must expose `coarse_logits`, `raw_detail_logits`,
`detail_logits`, and `combined_logits` in its trace. Unit tests require:

1. exact shape preservation;
2. global logits constant within each `8x8` block;
3. block-mean detail magnitude below `1e-6` in FP32;
4. combined logits equal coarse plus detail within `1e-6`;
5. zero-field and shuffled-field interventions change only the field input;
6. no gradient from the field path reaches the frozen retina; and
7. checkpoint serialization preserves all outputs.

## Exact-Capacity Global-Detail Control

A matched control replaces the candidate's local `F[i,j]` vector with the same
normalized global vector `z` repeated at all 16 cells. The input projection,
learned positional field, style modulation, patch decoder, coarse branch, and
all parameter shapes remain identical. Thus the two arms have exactly equal
trainable parameter counts:

```text
candidate detail input:      F[:, :, i, j]
global-detail control input: z repeated at every (i, j)
```

The control is allowed to generate fine detail from global identity and learned
position. It never receives `F`. This is a scientific control, not an admissible
final ILM writer. Both arms train from scratch on the same training partition
with the same batches, initialization seed, optimizer, schedule, and update
count. Each arm is selected independently on development data. A final paired
development audit uses the same freshly rendered examples for both selected
checkpoints.

## Local Occlusion Intervention

The `4x4` field is divided into four non-overlapping `2x2` quadrants. Evaluation
zeros one quadrant at a time while holding `z`, style, and all other field cells
fixed. For quadrant `q`, let

\[
\Delta_q=|p_{\mathrm{correct}}-p_{\mathrm{occlude}(q)}|.
\]

The matching output quadrant is `16x16`. Define

\[
L_q=\frac{\sum_{(i,j)\in q}\Delta_q(i,j)}
{\sum_{i,j}\Delta_q(i,j)+10^{-8}}.
\]

Report mean output change and mean locality over all four quadrants. A uniform
or unrelated change has expected locality near area fraction `0.25`; a
topographic route must exceed the fixed gate below.

## Image-Only Complexity Strata

V20 reuses the V19 geometry equation and thresholds without recalibration:

\[
C(x)=\operatorname{mean}(b)
+0.5\left[\operatorname{mean}|\Delta_x b|
+\operatorname{mean}|\Delta_y b|\right]
+0.1\operatorname{mean}
\left[\operatorname{AvgPool}_{4\times4}(b)>0.05\right],
\]

where `b = 1[x >= 0.5]`.

- simple: `C < 0.24`, weight `1.00`;
- medium: `0.24 <= C < 0.35`, weight `1.25`;
- dense: `C >= 0.35`, weight `2.00`.

No character identity or text label enters this score.

## Training Objective

Both arms use the same final-image objective:

\[
\begin{aligned}
\mathcal L_{20}={}&
\mathcal L_{\mathrm{WBCE}}+
\mathcal L_{\mathrm{Dice}}+0.5\mathcal L_1+
0.25\mathcal L_{\mathrm{edge}}\\
&+0.05\mathcal L_{\mathrm{id}}+0.05\mathcal L_{\mathrm{NCE}}
+0.25\mathcal L_{\mathrm{field\ margin}}
+0.25\mathcal L_{\mathrm{zero\ margin}}
+0.25\mathcal L_{\mathrm{coarse}}.
\end{aligned}
\]

The candidate margins require correct-field mean pixel error to beat a
batch-shuffled field and a zero field by `0.05`. For the global-detail control,
these two terms are zero by definition because it has no field input. The
coarse term is binary cross entropy between `sigmoid(c)` and the exact `4x4`
average-pooled target. Complexity weights apply to all per-example topology
terms.

Fixed optimizer and run settings:

- AdamW, learning rate `3e-4`, weight decay `0.03`;
- 100-step warmup and cosine decay to `0.1` of the initial rate;
- gradient norm clip `1.0`;
- BF16 autocast, batch size `32`;
- `1,600` updates per arm;
- validation every `200` updates on 512 generated candidates;
- training seed `20260820`;
- one RTX 4090 per arm; and
- no hyperparameter change after the first non-smoke candidate run.

Smoke mode may run at most 20 updates and is permanently marked
non-evidentiary. A smoke checkpoint cannot resume into an evidence run.

## Partition And Selection

Partition salt: `retinal-topology-router-v20`.

The manifest is partitioned by the first 64 bits of
`sha256(salt + NUL + identifier)` with holdout fraction `0.06`; half of the
holdout is development and half is frozen. The exact receipt records train,
development, and frozen counts plus the SHA-256 of sorted frozen identifiers.
Neither arm may instantiate a frozen image during training or selection.

Candidate checkpoints are ranked by dense correct pixel F1, then overall
correct F1, then earlier step. A candidate is selection-eligible only if:

1. overall correct pixel F1 `> 0.66`;
2. dense correct pixel F1 `> 0.68`;
3. dense correct-minus-shuffled-field F1 `> 0.12`;
4. dense correct-minus-zero-field F1 `> 0.10`;
5. correct identity top-1 `> 0.70` and above both-inputs-shuffled top-1;
6. correct target cosine `> 0.82` and above both-inputs-shuffled cosine;
7. correct-versus-shuffled-field pixel L1 `> 0.08`;
8. correct-versus-zero-field pixel L1 `> 0.08`;
9. mean quadrant-occlusion pixel change `> 0.02`;
10. mean quadrant-occlusion locality `> 0.40`;
11. semantic-reference versus target pixel L1 `> 0.05`;
12. global coarse within-block standard deviation `< 1e-6` in FP32;
13. detail block-mean absolute value `< 1e-6` in FP32; and
14. no frozen image is instantiated.

The global-detail control is selected by dense pixel F1 subject to overall F1
`>0.66`, dense F1 `>0.68`, identity top-1 `>0.70`, target cosine `>0.82`, the
same non-copying/frozen checks, and the same coarse/detail algebraic invariants.

After both arms are selected, the paired automatic gate additionally requires:

1. candidate dense F1 exceeds global-detail-control dense F1 by `>0.03`;
2. candidate overall F1 is no more than `0.02` below the control; and
3. candidate trainable parameters equal control trainable parameters exactly.

If no candidate checkpoint passes its arm-specific gates, no post-hoc paired
comparison can promote it.

## Blinded Readability Gate

Only after the candidate and paired automatic gates pass, generate one fixed
48-example development audit: 16 simple, 16 medium, and 16 dense outputs. Show
each output beside eight randomly ordered different-font visual candidates,
without text labels or target rows. Score visually duplicate groups using the
frozen retina's precomputed duplicate mask.

Required recognition remains:

- simple: at least `14/16`;
- medium: at least `12/16`;
- dense: at least `10/16`; and
- total: at least `38/48`.

Answers and keys are stored separately and scored by code. Examples may not be
replaced after review.

## Frozen Policy

The V19 frozen bank remains sealed and is never reused. V20 has a new salted
frozen partition. Frozen evaluation is forbidden until the candidate automatic
gate, matched-control gate, and blinded readability gate all pass. If any gate
fails, V20 is documented as development-only and its frozen images remain
uninstantiated.

## Claims

After development and matched-control success, allowed claims are limited to:

- local continuous retinal fields causally carry fine writing topology under
  the named V20 intervention;
- the field-primary route improves dense development topology over an
  exact-capacity global-detail control; and
- measured one-GPU parameters, memory, throughput, and quality.

Even after those gates, forbidden claims include autonomous language
generation, general understanding, historical factuality, Qwen parity, or
efficiency superiority. Those require predicted rather than supplied fields,
a readable closed write-reread loop, and the later Visual Word-Origin Book
benchmark.
