# Visual Cell Stream V25: Frozen Protocol

Date: 2026-08-13

Status: preregistered before implementation, optimization, or result inspection

## 1. Question

Can ordinary Chinese writing be learned as an ordered volume of visible images,
without a character vocabulary, and can a compact model use more than the last
visible cell to predict, draw, and reread the next cell?

V25 is the first post-V24 experiment on natural book language. It deliberately
does not require oracle, bronze, seal, or other historical forms. Those forms
remain future out-of-distribution capability tests. The immediate task is the
smaller and more important one: turn text into images, learn causal language
from those images, and produce the continuation as images.

The experiment is allowed to fail. Its result is not inspected until the code,
data split, metrics, controls, and thresholds below are fixed.

## 2. Native Representation

Each visible writing unit is a clean two-dimensional retinal observation. A
sentence is formed by stacking these observations along reading time:

\[
X\in[0,1]^{B\times T\times C\times H\times W},
\qquad C=1,\quad H=W=32.
\]

For one sample, dropping batch and channel axes gives the proposed
three-dimensional visual stream:

\[
X_{\mathrm{stream}}\in[0,1]^{T\times32\times32}.
\]

The third axis is ordered visual time, not geometric depth and not a character
ID. Each `32x32` slice is independently inspectable as writing. The sequence
can be packed into a page image for storage or display, but the model receives
the ordered stack. A future system may add a real depth axis; V25 fixes depth to
one because geometric 3D cannot answer the present language question.

`32x32` is the native prediction size. An `8x8` or `16x16` copy may be used as
an auxiliary coarse view, but never as the only generated record and never as
a color-encoded identity. Physical size, DPI, and centimeters are provenance
at the scanner or renderer boundary, not model inputs.

## 3. Student Boundary

Offline preparation may use strings, Unicode, OpenCC, OCR, LocalLLM, or another
open-source document tool. These tools may clean text, establish reading order,
create paired visual views, and audit generated output. They are absent from
the student batch and absent from deployed inference.

The deployed student may receive only:

- continuous grayscale cell images;
- their order in the visual stream;
- continuous flow time and random noise used by the image generator; and
- its own previously generated and reread cell images.

It may not receive:

- strings, bytes, token IDs, Unicode code points, character IDs, or vocabulary
  indices;
- OCR transcripts, teacher embeddings, text-model logits, or external model
  calls;
- color, alpha, depth, coordinates, or metadata that secretly encode identity;
- a finite glyph lookup table or nearest-character projection at inference;
- target labels, active lengths, or padding masks as language features; or
- a candidate image bank at deployed inference.

Alignment into equal retinal cells is sensor geometry. It does not make the
cells tokens: every value supplied to the student remains a measured pixel
intensity, and an unseen glyph image can occupy the same interface.

## 4. Data Contract

### 4.1 Evidence corpus

The initial evidence run uses
`data/visual_grammar/chinese_wikisource_public_domain.jsonl`, currently 7,017
records and about 2.90 million visible Han occurrences from 16 public-domain
Chinese works. The manifest and its provenance sidecar remain untracked data
artifacts. The evidence script must record their SHA-256 hashes, byte sizes,
record counts, rights strings, and source-title counts.

The preregistered local manifest SHA-256 is
`76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03`.
The evidence route refuses a different digest; exploratory runs must identify
themselves separately and cannot open the V25 frozen partition.

V25 may render both the original writing and an offline OpenCC simplified view.
Both views inherit the same record group and therefore cannot cross data
partitions. OpenCC output is preparation data, not a student-side conversion.
If OpenCC is unavailable, the evidence run uses original writing only and says
so in its receipt; it may not silently substitute another conversion.

Whitespace is not assigned an invisible ID. The first run omits whitespace and
retains visible punctuation. A later page model must represent spacing and line
breaks as actual blank geometry. Unsupported font glyphs are rejected by cmap
coverage before rendering so missing-glyph boxes cannot become a visual class.

### 4.2 Fixed record partition

Records are grouped by original manifest identifier before any script
conversion. Partition assignment is the first eight bytes of

```text
SHA256("ilm-v25-natural-chinese-cell-stream-20260813" + identifier)
```

mapped uniformly to `[0,1)`. The fixed intervals are:

- training: `[0.06, 1.00)`;
- development: `[0.03, 0.06)`; and
- frozen: `[0.00, 0.03)`.

The partition command may count identifiers without rendering frozen images.
Frozen strings, windows, labels, and images remain inaccessible until the
development gate authorizes one evaluation.

### 4.3 Font partition

Training uses only these installed faces:

- `NotoSansCJK-Regular.ttc`;
- `NotoSerifCJK-Regular.ttc`;
- `NotoSansCJK-Medium.ttc`; and
- `NotoSerifCJK-Light.ttc`.

Development uses `NotoSansCJK-Bold.ttc` and
`NotoSerifCJK-Medium.ttc`. Frozen evaluation uses
`NotoSerifCJK-Bold.ttc` and `NotoSansCJK-Light.ttc`. Font files and hashes are
recorded. Training font size, subpixel translation, contrast, and light blur
vary within fixed legibility bounds. Development and frozen views are
deterministic.

### 4.4 Training examples

Each example contains `65` consecutive visible cells. This produces 64 causal
next-cell transitions. Two independently rendered font views of the same source
span are created offline. Student-facing tensors are:

```text
context_view       float32 [64, 1, 32, 32]
next_view          float32 [64, 1, 32, 32]
independent_view   float32 [65, 1, 32, 32]
```

Names and metadata are host-side descriptions only. Before every model call a
boundary audit recursively rejects strings and integer tensors from the
student dictionary.

## 5. Model

V25 is a continuous visual-cell autoregressive model, not a classifier over
Chinese characters.

### 5.1 Retina

A convolutional retina maps each image to a continuous unit vector:

\[
z_t=\frac{R(x_t)}{\lVert R(x_t)\rVert_2},\qquad z_t\in\mathbb S^{191}.
\]

The evidence run initializes `R` from the image-only V16 retina and includes it
inside the final checkpoint. It is frozen in V25 so the causal experiment does
not rewrite the already measured cross-font sensor. A from-scratch option must
remain implemented, but is not the preregistered evidence route.

### 5.2 Causal visual field

Eight causal Transformer blocks of width 384 read the visual sequence. Rotary
position describes relative reading order; there is no learned character
embedding or vocabulary-sized matrix. Training corrupts earlier image cells at
independent continuous noise times so the model must tolerate its own imperfect
future generations:

\[
\tilde x_i=(1-s_i)x_i+s_i\epsilon_i,
\qquad s_i\sim U(0,s_{\max}).
\]

The field predicts a continuous next-retina proposal

\[
\hat z_{t+1}=P(h_t) / \lVert P(h_t)\rVert_2.
\]

It is trained against an independently rendered image view. Candidate labels
are never supplied to this path.

### 5.3 Continuous cell writer

For clean next-cell image `x`, Gaussian noise `epsilon`, and flow time
`tau in [0,1]`, define

\[
y_\tau=(1-\tau)x+\tau\epsilon,
\qquad u_\tau=\epsilon-x.
\]

A small convolutional flow writer predicts

\[
\hat u=W(y_\tau,\tau,h_t,\hat z_{t+1}).
\]

The writer has no glyph output table. At inference it integrates from noise at
`tau=1` to pixels at `tau=0`. Multiple continuous candidates may be sampled;
the model's own retina selects the candidate closest to `hat z`. The selected
pixels are appended to the stream and reread before the following prediction.

### 5.4 Objective

The fixed objective is

\[
\mathcal L=
\mathcal L_{\mathrm{visual}}+
0.50\mathcal L_{\mathrm{contrastive}}+
1.00\mathcal L_{\mathrm{flow}}+
0.20\mathcal L_{\mathrm{endpoint}}+
0.10\mathcal L_{\mathrm{reread}}.
\]

`L_visual` is cosine distance from `hat z` to the independent target retina.
`L_contrastive` distinguishes only in-batch image observations and uses
multi-positive visual similarity to avoid treating duplicate visible forms as
known negatives. `L_flow` is velocity MSE. `L_endpoint` is stroke-weighted
pixel loss on the predicted clean endpoint. `L_reread` compares the retina of
that endpoint with the independent target view. No term receives a character
label.

## 6. Fixed Evidence Run

The candidate configuration is:

| Property | Value |
|---|---:|
| context cells | 64 |
| cell shape | `1x32x32` |
| retina dimension | 192 |
| causal width / layers / heads | 384 / 8 / 6 |
| writer base channels | 48 |
| language-stage updates | 2,400 |
| writer-stage updates | 1,200 |
| effective batch | at least 24 streams |
| precision | BF16 |
| maximum allocated VRAM | 18 GiB |
| inference flow steps | 12 |
| candidates per generated cell | 4 |
| autonomous rollout | 16 cells |

The implementation may lower the physical microbatch and use gradient
accumulation. It may not change model width, depth, context, update counts,
partition salt, evidence fonts, or metric thresholds after seeing development
results. A smoke run may use at most 20 updates and cannot select a model.

The language stage freezes the retina and writer and optimizes the causal field
and proposal. The writer stage freezes the accepted language field and trains
only the continuous cell writer. This separation prevents a typographic loss
from being mistaken for language learning.

## 7. Development Evaluation

An evaluator-only visual bank contains the 1,024 most frequent supported Han
forms in the training partition, each rendered in development fonts. It is not
deployed. Target character strings are opened only inside the evaluator to map
the correct image row.

Report next-cell top-1, top-5, normalized log probability, and target cosine
for:

- full 64-cell visual history;
- last visible cell only;
- order-shuffled history with the final cell preserved;
- blank history;
- image-unigram frequency; and
- a symbolic bigram computed offline from training text.

Also report context-paired counterfactuals in which two examples have the same
last visible cell but different histories and different true next cells. This
tests whether output changes for the correct visual reason.

Writer evaluation reports sampled target identity through the held-out visual
bank, target cosine after rereading, pixel F1 against an independent font view,
blank rate, repeated-cell rate, and autonomous 16-cell degradation. OCR may be
reported as an evaluator-only diagnostic but cannot determine selection.

## 8. Selection Gates

The language model selects only if all are true on a fresh 2,048-window
development audit:

- full-history top-1 exceeds last-only by more than `0.03`;
- full-history top-1 exceeds image unigram by more than `0.03`;
- full-history normalized target log probability exceeds last-only by more
  than `0.05` nat;
- order-shuffled top-1 is below full-history top-1 by more than `0.015`;
- counterfactual switch accuracy exceeds `0.55`;
- full-history target cosine exceeds `0.55`;
- the inference-boundary audit passes; and
- peak allocated VRAM is below 18 GiB.

Symbolic bigram is a required benchmark but not a V25 selection gate. Failure
to beat it is reported plainly and blocks any broad language-model claim.

The writer selects only after the language model selects and all are true:

- four-sample generated identity top-1 exceeds `0.20` over the 1,024-image
  bank;
- reread target cosine exceeds `0.60`;
- generated pixel F1 exceeds `0.45`;
- blank rate is below `0.05`;
- autonomous position-16 ink density remains within `[0.50, 1.50]` times the
  teacher-forced target density; and
- every generated continuation is formed by appending and rereading actual
  output pixels.

Only if both stages select is one frozen evaluation authorized. Failure is a
useful result and does not permit threshold revision, additional checkpoint
selection, or frozen inspection.

## 9. Claims Allowed

A passing V25 may claim a compact model learned a bounded natural-Chinese
next-cell dependency and generated short continuations through an image-only
runtime. It may not claim Qwen parity, arbitrary question answering, human-like
reading, universal script understanding, better efficiency than LLMs, or
historical-form knowledge.

This protocol is informed by PIXAR's demonstration of autoregressive rendered
text patches and its observed exposure-bias/readability failure, and by xAR's
continuous next-entity flow with noisy-context training. V25 differs by using
inspectable Chinese glyph images directly, no VQ/VAE code sequence, a strict
image-only deployed boundary, and mandatory generated-pixel rereading:

- <https://arxiv.org/abs/2401.03321>
- <https://arxiv.org/abs/2502.20388>
