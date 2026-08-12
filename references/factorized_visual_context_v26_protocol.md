# Factorized Visual Context V26: Frozen Protocol

Date: 2026-08-13

Status: preregistered before V26 implementation, optimization, smoke training,
or result inspection

Research rationale:
[`factorized_visual_context_v26_research.md`](factorized_visual_context_v26_research.md)

## 1. Fixed Question

Can a compact, image-only Chinese model use visual history earlier than a
pixel-identical four-glyph suffix to improve the conditional distribution of
the next glyph image?

V26 is a language-state experiment. A pixel writer is forbidden until every
language gate in Section 9 passes. Frozen examples remain sealed until the
same gate authorizes one evaluation.

## 2. Representation And Student Boundary

One observation is a continuous grayscale image

\[
x_t\in[0,1]^{1\times32\times32}.
\]

A context is an ordered visual-time volume

\[
X_t=(x_{t-63},\ldots,x_t)
\in[0,1]^{64\times1\times32\times32}.
\]

The stream may be packed by the existing reversible serpentine lattice for
display. Packing cannot alter cell pixels or causal order and is not part of
V26 optimization.

Every student call may contain only floating image tensors, continuous noise,
and model states derived from those tensors. It may not contain strings,
bytes, token IDs, Unicode/code-point values, character IDs, vocabulary
indices, OCR output, external language-model states, source identifiers,
script labels, glyph lookups, or an inference candidate bank.

Offline preparation and evaluator code may use strings to establish reading
order, record-group partitions, suffix equality, targets, and baselines. A
recursive boundary assertion must reject non-floating student values.

## 3. Fixed Data

### 3.1 Corpus and partitions

Use only
`data/visual_grammar/chinese_wikisource_public_domain.jsonl`, SHA-256
`76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03`.
Reuse the V25 record-group salt and partitions exactly:

```text
SHA256("ilm-v25-natural-chinese-cell-stream-20260813" + identifier)
train       [0.06, 1.00)
development [0.03, 0.06)
frozen      [0.00, 0.03)
```

Reusing the split makes V25 and V26 development diagnostics comparable. It
does not permit V26 to inspect V25 frozen strings or images before selection.
Original and offline OpenCC-simplified views inherit the original record's
partition.

Reuse the V25 train, development, and frozen font partitions and record every
font hash. Unsupported cmap entries and whitespace remain excluded. Visible
punctuation remains part of the image stream, but the fixed evaluator target
bank contains the 1,024 most frequent supported Han forms from training.

### 3.2 Natural endpoint examples

Each natural sample contains 72 consecutive visible cells:

```text
context             float32 [64, 1, 32, 32]
future              float32 [ 8, 1, 32, 32]
reference_context   float32 [64, 1, 32, 32]
reference_future    float32 [ 8, 1, 32, 32]
```

The two views use independently chosen training fonts and augmentations. A
training call predicts future offsets `1, 2, 4, 8`; it does not average losses
over shorter prefixes.

### 3.3 Causal pair examples

The fixed training sampler builds 16,384 natural pair records from the
training partition. Pair members must:

- end in exactly the same four visible characters;
- have different next visible characters;
- come from different original record identifiers; and
- contain 64 context cells and at least one target cell.

One pair is retained per suffix key using deterministic SHA-256 priority with
seed `20260910`. Pair contexts are rendered using identical font and
augmentation draws, making all four shared suffix cells pixel-identical. A
second font renders independent target observations.

The development audit fixes 512 cross-record suffix-4 pairs with seed
`20260911`. A separate 512-pair suffix-8 diagnostic may use members from the
same record because only 59 cross-record suffix-8 keys exist; it is not a
selection gate.

Pair strings and source IDs stay in host metadata. The student sees eight
floating tensors: two contexts and two next images, each in two independent
visual views.

## 4. Fixed Model

### 4.1 Retina

Initialize and freeze the V16 image retina from

```text
artifacts/predictive_visual_field_v16_memory_pilot/checkpoint_step_0002200.pt
```

with expected SHA-256
`90791001203640f0de66316cf2e30b3e2c588480fef0e3d9d4f6283ba043ecbe`.
The model checkpoint contains the retina. A from-scratch retina is exploratory
and cannot produce V26 evidence.

### 4.2 Factorized context

For context `X=(x_1,...,x_64)`, the appearance branch receives only
`R(x_64)`. The history branch receives only
`(R(x_1),...,R(x_63))`. Eight width-384, six-head causal visual blocks with
rotary reading position produce the history state. The fused state is

\[
s=\operatorname{RMSNorm}(A(R(x_{64}))+\sigma(G)\odot C(H(X_{1:63}))).
\]

`G` is context-dependent and continuous. No learned sequence token, character
embedding, or vocabulary matrix is allowed. For a one-cell context the
history residual is exactly zero.

### 4.3 Conditional particles

For each horizon in `{1,2,4,8}`, eight independent Gaussian noise vectors of
dimension 64 enter a shared conditional proposal MLP. It emits eight
unit-normalized 192-dimensional retina particles. Particles are conditional
samples, not persistent glyph prototypes. Evaluation uses a fixed seeded
Gaussian draw; training resamples noise.

The fixed architecture is:

| Property | Value |
|---|---:|
| context cells | 64 |
| future cells available | 8 |
| cell shape | `1x32x32` |
| retina dimension | 192 |
| model width / layers / heads | 384 / 8 / 6 |
| MLP ratio | 3.0 |
| dropout | 0.05 |
| future horizons | `1,2,4,8` |
| particles per horizon | 8 |
| particle noise dimension | 64 |
| maximum trainable parameters | 20 million |

## 5. Fixed Scores And Losses

Let `Q_h={q_hk}` be the eight particles for horizon `h`, and let `y_h` be the
independently rendered target retina. Distance is Euclidean distance on the
unit sphere. The energy score is

\[
\mathcal E_h=\frac18\sum_k d(q_{h,k},y_h)
-\frac1{128}\sum_{k,j}d(q_{h,k},q_{h,j}).
\]

Candidate image state `v` receives continuous score

\[
S(Q_1,v)=-\frac18\sum_k d(q_{1,k},v).
\]

The contrastive loss uses this score against current targets plus an 8,192
entry FIFO queue of detached target-retina observations. Targets with retinal
cosine at least `0.985` are multiple positives. The queue is training-only,
contains no IDs, and is absent at deployed inference.

For a suffix-matched pair `(A,y_A),(B,y_B)`, define

\[
\mathcal L_{pair}=\frac12\left[
\operatorname{softplus}(0.10-S(Q_A,y_A)+S(Q_A,y_B))+
\operatorname{softplus}(0.10-S(Q_B,y_B)+S(Q_B,y_A))
\right].
\]

The fixed total per update is

\[
\mathcal L=
1.00\mathcal E_1+
0.50\frac{\mathcal E_2+\mathcal E_4+\mathcal E_8}{3}+
1.00\mathcal L_{contrastive}+
1.00\mathcal L_{pair}.
\]

Natural contexts receive independent continuous corruption times uniformly in
`[0,0.15]`. Pair contexts remain clean. No character-label loss is permitted.

## 6. Fixed Optimization

| Property | Value |
|---|---:|
| optimizer | AdamW `(0.9, 0.95)` |
| updates | 8,000 |
| natural pair records per microbatch | 8 |
| suffix-pair records per microbatch | 4 |
| bidirectional visual views | yes |
| gradient accumulation | 4 |
| base learning rate | `3e-4` |
| warmup | 400 updates |
| minimum LR ratio | 0.10 |
| weight decay | 0.05 |
| gradient clipping | 1.0 |
| precision | BF16 |
| random seed | `20260909` |
| natural dataset seed | `20260910` |
| pair seed | `20260910` |
| maximum allocated VRAM | 18 GiB |

The effective update contains 64 natural view-contexts and 64 pair
view-contexts across accumulation. Checkpoints, optimizer state, source hashes,
protocol hash, corpus receipt, fonts, parameter shapes, peak VRAM, and boundary
receipt are recorded. A smoke run may use at most 20 updates and is never
evidence.

No V25 language weights initialize V26. Only the preregistered frozen V16
retina transfers.

## 7. Development Evaluation

Use 2,048 deterministic natural development windows and the evaluator-only
1,024-form, two-font visual bank. Report top-1, top-5, normalized target log
probability, target score, target cosine to the nearest proposal particle, and
energy score for:

- full 64-cell history;
- suffix lengths `1,2,4,8,16,32,64`;
- earlier history zeroed while the final four cells remain unchanged;
- cells 1--60 shuffled while the final four remain unchanged; and
- the factorized history residual swapped between suffix-matched pair members.

Report image-unigram and training-text symbolic bigram top-1, top-5, and target
log probability. These are evaluator baselines, never student inputs.

For the 512 cross-record suffix-4 pairs, report:

- two-way pair ranking accuracy;
- mean correct-versus-other score margin;
- output switch rate between pair contexts;
- swapped-residual target accuracy; and
- proof that the four suffix image tensors are exactly equal.

Report the same metrics for the suffix-8 diagnostic without using them for
selection. Also report particle spread, target coverage by the best particle,
retina-bank oracle accuracy, parameter count, throughput, peak VRAM, and every
student-boundary flag.

## 8. Causal Interventions

The model API must expose appearance state, history residual, and conditional
particle prediction as separate operations. The evaluator constructs:

```text
normal A: appearance(A) + history(A)
normal B: appearance(B) + history(B)
swap A:   appearance(A) + history(B)
swap B:   appearance(B) + history(A)
last only: appearance(.) + 0
```

Suffix-matched appearance pixels are equal. A successful residual swap must
move preference toward the target associated with the donated history. Merely
changing output under a swap does not count; the direction must be correct.

## 9. Frozen Selection Gates

All mechanism gates must pass on the fixed development audit:

- full-history top-1 exceeds last-only top-1 by more than `0.02`;
- full-history top-1 exceeds image unigram by more than `0.03`;
- full-history target log probability exceeds last-only by more than `0.10`
  nat;
- full-history target log probability exceeds suffix-4 by more than `0.03`
  nat;
- full-history target log probability exceeds suffix-4-preserving shuffled
  history by more than `0.03` nat;
- cross-record suffix-4 two-way pair ranking exceeds `0.65`;
- swapped-residual target accuracy exceeds `0.65`;
- every audited suffix-4 image equality check passes;
- retina-bank oracle top-1 is at least `0.99`;
- the image-only boundary audit passes; and
- peak allocated VRAM is below 18 GiB.

The model is language-qualified only if both additional gates pass:

- full-history top-1 exceeds symbolic bigram top-1 by more than `0.01`; and
- full-history target log probability exceeds symbolic bigram target log
  probability by more than `0.05` nat.

Only a language-qualified model may authorize one V26 frozen evaluation and a
later preregistered pixel-writer stage. Passing mechanism gates but failing the
bigram gates is evidence for bounded visual context use, not a useful language
model. Thresholds cannot be changed after a development result.

## 10. Allowed Claims

If only mechanism gates pass, V26 may claim a compact image-only model learned
a bounded next-glyph dependency beyond a shared four-glyph visual suffix.

If every gate passes, V26 may additionally claim that its bounded development
distribution outperformed fixed unigram and bigram controls while using only
image tensors at the student boundary.

Neither outcome supports claims of general question answering, human-like
reading, arbitrary historical-form synthesis, OCR replacement, Qwen parity,
book-scale modeling, or superior efficiency to token LLMs. Those require later
preregistered experiments and direct matched baselines.
