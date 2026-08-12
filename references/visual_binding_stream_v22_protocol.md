# Visual Binding Stream V22 Protocol

Date preregistered: 2026-08-13

Status: fixed before V22 implementation or training

## Question

V16 shows that a compact image-only recurrent state uses visual history, but it
does not beat the symbolic bigram. V21 shows that a continuous local retinal
field can carry a complete spatial writing plan, but its disjoint patch writer
does not select. Neither experiment tests whether a rendered Chinese prompt can
control a generated answer image.

V22 asks the smallest prompt-following question that also exercises arbitrary
visual form:

> Can an image-only student read a short Chinese relational prompt, bind two
> previously unseen glyph images to visible labels, and generate the requested
> glyph as a new image, changing its output when only the visual query changes?

This is a bounded visual-binding proof, not open-domain question answering. It
is chosen because a model that cannot solve controlled prompt binding cannot
credibly solve book continuation, etymology, or free-form image answers. A
positive result permits a longer Visual Language Stream experiment; it does not
establish general language understanding.

## Relation To Primary Work

- [PIXAR](https://arxiv.org/abs/2401.03321) demonstrates autoregressive language
  input and output in rendered pixel patches, and reports that maximum-
  likelihood nonoverlapping patch generation often becomes noisy. V22 keeps
  the image-only boundary but tests Chinese glyph frames, explicit prompt
  counterfactuals, continuous retinal states, and overlapping local output.
- [Visual Prompting via Image Inpainting](https://arxiv.org/abs/2209.00647)
  demonstrates that input/output examples can define an image-to-image task.
  V22 instead uses visible Chinese relational instructions and generates a new
  answer frame.
- [Slot Abstractors](https://proceedings.mlr.press/v235/mondal24a.html) and
  [Parallelized Spatiotemporal Slot Binding](https://proceedings.mlr.press/v235/singh24g.html)
  support explicit relational binding and parallel attention over visual
  objects or time. V22 uses a much smaller supervised visual-memory test and
  does not claim object-centric discovery.

## Visual Language Stream Interface

The student interface is

\[
X_{\mathrm{prompt}}\in[0,1]^{B\times T_p\times1\times32\times32}
\longrightarrow
\hat Y_{\mathrm{answer}}\in[0,1]^{B\times T_a\times1\times32\times32}.
\]

V22 fixes `T_p=6`, `T_a=1`, and `D=1`. The implementation must retain the
answer-time dimension so later experiments can use `T_a>1` without changing
the public tensor contract. Calling V22 a movie model, page model, or
open-domain ILM is forbidden.

## Episode Grammar

Every prompt is a six-frame visual stream:

```text
[label_1, glyph_1, label_2, glyph_2, operation, query_label]
```

The label pairs are:

```text
(甲,乙), (天,地), (上,下), (左,右)
```

The operations are:

- `同`: output the glyph bound to `query_label`;
- `异`: output the glyph bound to the other visible label.

The two label/glyph pairs are randomly ordered. `glyph_1` and `glyph_2` are
distinct. The query label is sampled uniformly. The paired counterfactual keeps
all five preceding frames fixed and changes only the final query-label image;
the correct answer must switch.

No operation, label, slot index, character identity, or answer index enters the
student. Those values exist only in the offline renderer and evaluator. The
student receives the six image tensors. The target is a separately rendered
image tensor in one fixed canonical output face. Source glyph frames use
independent noncanonical faces and augmentations, preventing exact pixel paste
from satisfying the canonical target loss.

## Identity And Composition Split

Build a bank from the `1,024` most frequent Han characters in
`data/visual_grammar/chinese_wikisource_public_domain.jsonl` that are supported
by every registered retinal font. Exclude every label/operation character from
the candidate bank.

Identity split salt: `visual-binding-stream-v22`.

For each candidate character, use the first 64 bits of
`sha256(salt + NUL + character)`:

- fraction `<0.80`: training identity;
- fraction `0.80..0.90`: development identity;
- fraction `>=0.90`: frozen identity.

Development and frozen targets therefore contain no glyph identity used as a
training target or source candidate. Marker glyphs are shared task language,
not answer candidates.

The implemented deterministic partition resolves to:

- `815` training identities;
- `104` development identities, identifier SHA-256
  `86007f870644707c6de2379f068c2ac5666265661891e0aa1ed964ed13815047`;
  and
- `105` frozen identities, identifier SHA-256
  `7e144212e1b90a64cd5b7ad095ed2b95ccd6aa52095b2ae474d30cfec5a438de`.

Two operation/label-pair combinations are withheld from training while keeping
every operation and label pair visible elsewhere:

```text
(同, 天/地), (异, 左/右)
```

Development reports seen-combination and held-out-combination strata
separately. Frozen identities and rendered frozen images remain inaccessible
until every automatic, paired, and blinded gate passes.

## Student Boundary

The learned path may receive only:

- six grayscale prompt frames;
- continuous states produced from those frames by the frozen image retina; and
- generated image frames reread by the same retina during training or rollout.

It may not receive strings, token IDs, Unicode IDs, OCR, character labels,
operation IDs, slot indices, target indices, a finite visual codebook, glyph
lookup, candidate embeddings, evaluator scores, or an external language model.
Typed/source strings exist only in the offline renderer and are deleted before
the model call. Candidate images and identity metadata are evaluator-only.

The frozen retina checkpoint is:

```text
artifacts/predictive_visual_field_v16_memory_pilot/checkpoint_step_0002200.pt
SHA-256 90791001203640f0de66316cf2e30b3e2c588480fef0e3d9d4f6283ba043ecbe
```

No V17--V21 writer weight may enter V22. All trainable binding and writing
weights start from the fixed V22 seed.

## Architecture

### Continuous visual binder

The frozen retina maps each prompt frame to global and local continuous state:

\[
(z_t,F_t)=R(x_t),\qquad
z_t\in\mathbb R^{192},\quad
F_t\in\mathbb R^{192\times4\times4}.
\]

A four-block, eight-head, pre-normalized visual Transformer with width `256`,
feed-forward width `768`, and dropout `0.05` reads the six global states plus
learned sensory-time positions. It has no token embedding or output
vocabulary. The last visual state forms a continuous query over all six frame
states:

\[
h_{1:6}=\mathcal T(P(z_{1:6})+p_{1:6}),\qquad
\alpha_t=\operatorname{softmax}_t
\left(q(h_6)^\top k(h_t)/\sqrt{256}\right).
\]

Shared continuous value maps produce a bound global/local plan:

\[
\hat z=\sum_t\alpha_t V_z(z_t),\qquad
\hat F=V_F\left(\sum_t\alpha_t F_t\right).
\]

No supervised attention index is allowed. Target images supervise the selected
state and generated pixels. Attention mass is diagnostic only.

### Overlapping local writer

V21 emitted disjoint `8x8` patches. V22 instead emits one `12x12` patch from
each `4x4` local cell at stride `8`, padding `2`, and combines them with a fixed
positive partition-of-unity window:

\[
\ell=
\frac{\operatorname{Fold}_{12,8,2}(w\odot A(\hat F,\hat z))}
{\operatorname{Fold}_{12,8,2}(w)+\epsilon},
\qquad \hat y=\sigma(\ell).
\]

The local decoder uses shared weights, spatially uniform global modulation, and
at most one replicate-padded `3x3` field-neighborhood operation. It receives no
coordinate map or global spatial seed. Unit tests must measure the exact input
cell influence radius and verify that a constant tiled field cannot acquire a
content-specific global spatial plan.

The complete trainable student contains exactly `3,410,128` parameters, below
the fixed `4,000,000` limit. Candidate and query-blind control have exactly
equal trainable parameter counts and parameter shapes. This count is fixed by
structural tests before the first smoke or evidence run and may not be changed
afterward.

### Query-blind control

The matched control replaces only the final query-label retinal state with a
learned continuous null state before the same Transformer. Every module,
parameter, operation, optimizer setting, batch, target, schedule, and update
count remains identical. The null global and field parameters exist in both
arms and are initialized from the fixed training seed with a zero-mean Gaussian
of standard deviation `0.02`; nonzero initialization avoids the singular
gradient of normalizing the origin. The control therefore sees both labeled
glyph bindings and the operation but cannot know which query label was shown.

## Training Objective

Each batch contains the correct and paired-counterfactual prompt. The same
model processes both. Let `y` be the canonical target image, `(z_y,F_y)` its
frozen retinal state, and `(z_g,F_g)` the state reread from generated pixels.

\[
\begin{aligned}
\mathcal L_{22}={}&
\mathcal L_{\mathrm{WBCE}}+\mathcal L_{\mathrm{Dice}}
+0.5\mathcal L_1+0.25\mathcal L_{\mathrm{edge}}\\
&+0.20(1-\cos(\hat z,z_y))
+0.15\mathcal L_{\mathrm{field}}(\hat F,F_y)\\
&+0.10(1-\cos(z_g,z_y))
+0.10\mathcal L_{\mathrm{field}}(F_g,F_y)
+0.05\mathcal H(\alpha).
\end{aligned}
\]

The target image supplies supervision; there is no identity classification
head. A separate oracle-writer auxiliary pass may condition the same writer on
the selected source glyph image, but it may receive no symbolic target index.
Its loss weight is fixed at `0.50` and its metrics are reported separately so a
binding failure cannot be misreported as a writing failure.

## Fixed Optimization

- AdamW, learning rate `3e-4`, betas `(0.9,0.95)`, weight decay `0.03`;
- 100-step warmup and cosine decay to `0.1` of the initial rate;
- gradient norm clip `1.0`;
- BF16 autocast, batch size `64`;
- exactly `1,600` updates per arm;
- validation every `200` updates on `512` paired development episodes;
- training seed `20260822` and dataset seed `20260823`;
- one RTX 4090 per arm, trained sequentially; and
- no threshold, architecture, data split, or hyperparameter change after the
  first non-smoke candidate run.

Smoke mode may run at most 20 updates and is permanently non-evidentiary. A
smoke checkpoint cannot resume into an evidence run.

## Structural Tests

Before evidence training, tests must require:

1. prompt and answer stream tensor shapes, including answer-time dimension;
2. no integer or string tensor enters model `forward`;
3. frozen retina receives no gradient;
4. candidate/control parameter-shape equality and fewer than 4M parameters;
5. changing only the final query frame changes the candidate query state;
6. changing the final query frame has exactly zero effect in the control;
7. attention weights are finite, normalized, and cover six visual frames;
8. a one-cell field intervention has a bounded measured output support;
9. overlap weight normalization is positive over every output pixel;
10. a constant tiled field remains content-spatially uniform under the local
    decoder before fixed overlap/cropping effects;
11. generated pixels backpropagate to query and source frames in the candidate;
12. checkpoint serialization preserves configuration and boundary receipt; and
13. no frozen image is instantiated by dataset construction or validation.

## Development Metrics And Candidate Selection

The evaluator builds an image-only bank from development identities and
independent font views. Identity labels are evaluator-only. Rank eligible
candidate checkpoints by counterfactual switch accuracy, full-model global
identity top-1, pixel F1, then earlier step. Eligibility requires all of:

1. paired binary-choice accuracy `>0.85`;
2. counterfactual switch accuracy `>0.80`;
3. held-out operation/label-combination switch accuracy `>0.75`;
4. global identity top-1 over at least 96 unseen development identities
   `>0.45`;
5. global identity top-1 exceeds query-shuffled top-1 by `>0.20`;
6. generated target cosine `>0.78`;
7. generated overall pixel F1 `>0.58`;
8. oracle-writer overall pixel F1 `>0.64`;
9. mean paired-output pixel L1 after changing only the query `>0.08`;
10. generated target cosine exceeds both operation-frame cosine and final
    query-label cosine by `>0.15`;
11. student-boundary receipt is clean; and
12. frozen images instantiated equals zero.

Binary-choice accuracy asks whether generated output is closer, in frozen
retinal cosine, to the correct source identity than the distractor identity.
Counterfactual switch accuracy requires both members of a paired episode to
choose their respective answers. These metrics test prompt dependence without
feeding identity labels to the student.

## Control And Paired Gate

The query-blind control selects its best structural checkpoint by binary-choice
accuracy, then pixel F1, then earlier step, without a quality minimum. On one
fresh paired development audit:

1. candidate switch accuracy must exceed control by `>0.25`;
2. candidate global identity top-1 must exceed control by `>0.20`;
3. candidate paired-output pixel L1 must exceed control by `>0.06`;
4. candidate must retain every arm-specific gate;
5. control must retain every structural and sealed-split gate; and
6. parameter counts and shapes must be exactly equal.

No endpoint comparison may replace selected checkpoints. The fresh paired
audit is fixed to `1,024` paired development episodes, batch size `64`, four
independent noncanonical identity-bank views, and audit seed
`dataset_seed + 2,000,003`.

## Blinded And Frozen Policy

Only after automatic candidate and paired gates pass, create a fixed 48-pair
development review. Reviewers see prompt frames and two generated answer images
for the counterfactual queries, without text transcription or answer labels.
They choose which visible source form each answer depicts. Required paired
accuracy is at least `40/48`, including at least `10/12` held-out-combination
pairs.

Only after that numeric blinded gate passes may a separate evaluator instantiate
the frozen identity images once. It must verify protocol, split, font,
checkpoint, and prior-gate hashes before rendering.

## Allowed And Forbidden Claims

After development success only, the allowed claim is:

- a compact image-only student follows the named visual binding grammar and
  generates requested unseen Chinese forms under query counterfactuals.

Even after frozen success, forbidden claims include open-domain understanding,
free-form Chinese generation, factual etymology, book-level language modeling,
arbitrary historical-form synthesis, movie generation, Qwen parity, or
efficiency superiority. Those require longer answer streams, public-domain
language pretraining, provenance-aware historical episodes, and task-level
comparisons.
