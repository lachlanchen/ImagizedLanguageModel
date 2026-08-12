# Visual Packet Reread Stream V24 Protocol

Date preregistered: 2026-08-13

Status: fixed before V24 implementation, smoke training, or evaluation

## Question And Claim Boundary

Can a compact image-only student locate roles from visible packet headers in a
variable-length prompt, execute the V23 same/other relation, emit one unseen
Chinese glyph image, reread that generated image, and emit its visibly bound
label as a second image?

After development success only, the maximum claim is that the student follows
this fixed packet grammar and generates a causal two-frame image stream on
unseen development identities. Frozen success extends that statement only to
the sealed identity split. Open-domain language understanding, arbitrary
sentences, factual etymology, historical-form synthesis, book continuation,
page/movie generation, Qwen parity, and efficiency superiority remain
forbidden claims.

## Image-Only Stream Contract

The model input is

\[
X\in[0,1]^{B\times T\times1\times32\times32},
\qquad T\bmod3=0,
\]

and its autonomous output is

\[
\hat Y\in[0,1]^{B\times2\times1\times32\times32}.
\]

Each consecutive group of three frames is a packet
`[header, content_A, content_B]`:

```text
[契, label, glyph]       binding packet, exactly two
[法, 同 or 异, blank]    operation packet, exactly one
[问, label, blank]       query packet, exactly one
[旁, distractor, blank]  distractor packet, zero to three
[止, blank, blank]       end packet, exactly one and last
```

Packets before `止` are uniformly shuffled. Every frame is independently
rendered in a noncanonical face. Blank frames contain zeros. A collator pads to
the longest stream in a batch using all-zero three-frame packets. The model
receives neither active length nor a padding mask.

Training prompts contain `5`, `6`, or `7` active packets (`T=15,18,21`) with
zero, one, or two distractors. Development contains those lengths and a sealed
held-out length of `8` packets (`T=24`) with exactly three distractors. The
implementation must accept every packet-aligned `T` from `15` through `24`.

Label pairs are `甲/乙`, `天/地`, `上/下`, and `左/右`. Operations are `同`
and `异`. `同` selects the glyph bound to the query label; `异` selects the
other glyph. Answer frame 1 is that glyph in the canonical output face. Answer
frame 2 is the canonical image of the label visibly bound to the glyph emitted
in frame 1.

## Counterfactuals

Each rendered episode also supplies:

1. a query counterfactual changing only the query packet's `content_A` image;
2. an operation counterfactual changing only the operation packet's
   `content_A` image;
3. a packet permutation preserving every frame inside each packet;
4. a distractor counterfactual that inserts one distractor when fewer than
   three exist, or changes only one distractor image at maximum length;
5. a teacher-forced frame-1 image using the canonical target; and
6. a history intervention replacing frame 1 at the reread boundary with the
   other source glyph's canonical image.

The corresponding image targets are rendered independently. Strings and
metadata are deleted before every student call.

## Identity And Composition Split

Build the `1,024` most frequent supported Han identities from
`data/visual_grammar/chinese_wikisource_public_domain.jsonl`. Exclude all label,
operation, and packet-header characters:

```text
甲 乙 天 地 上 下 左 右 同 异 契 法 问 旁 止
```

Identity salt: `visual-packet-reread-stream-v24`.

Partition each character by the first 64 bits of
`sha256(salt + NUL + character)`:

- `<0.80`: training;
- `0.80..0.90`: development; and
- `>=0.90`: frozen.

The identifier-only receipt, computed without rendering development or frozen
images, is fixed as:

- training identities: `829`;
- development identities: `88`;
- development identifier SHA-256:
  `2b611e66778061319bb2502ad850c635b5d89e81e9eab7f9c8ef23a09514e892`;
- frozen identities: `107`; and
- frozen identifier SHA-256:
  `d3f6d51ef6c0cb0eeeab664d89e8a2c467bc35ea7482ae55f873f4b28b85c2ab`.

The V24 held-out operation/label combinations remain `(异, 天/地)` and
`(同, 左/右)`. Every component occurs separately in training.

## Frozen Foundations

The retina is the selected V16 checkpoint:

```text
artifacts/predictive_visual_field_v16_memory_pilot/checkpoint_step_0002200.pt
SHA-256 90791001203640f0de66316cf2e30b3e2c588480fef0e3d9d4f6283ba043ecbe
```

The visual writer is the selected V23 canonicalizer:

```text
artifacts/visual_canonicalizer_v23_evidence/checkpoint_selected_development.pt
SHA-256 26cf1bab490abe867e7055a679eff6a9e26e81ad78e6cd9694afd3e425c06135
```

The operation reader and visual-match temperature come from the selected V23
relation-aware checkpoint:

```text
artifacts/visual_relation_circuit_v23_evidence/relation_aware/
checkpoint_selected_development.pt
SHA-256 69c5cb06a5f02b5bed26b8687042534e9481fec96bea6ab41e2e00df7c23df43
```

Every inherited parameter is frozen. V24 may not load any other V17--V23 binder
or writer weight.

## Student Boundary

The learned path may receive only prompt images, frozen-retina states computed
from those images, routed continuous source pixels, generated answer pixels,
and frozen-retina states obtained by rereading generated pixels. It may not
receive strings, token IDs, Unicode IDs, OCR, role labels, packet indices,
active lengths, padding masks, character/operation labels, target indices, a
visual codebook, glyph lookup, evaluator scores, or an external language model.

The architecture knows only that consecutive triples form packets and that
their relative offsets are header/content/content. It has no absolute frame or
packet positions.

## Architecture

For packet `j`, encode header and contents as normalized frozen-retina states
`H_j,A_j,B_j`. Three learned normalized prototypes score the headers:

\[
e_{jr}=\tau_r\langle H_j,p_r\rangle,
\qquad r\in\{pair,operation,query\}.
\]

Each `tau_r` is a learned positive scalar initialized to `8.0` and bounded to
`[1,32]`. Straight-through hard routing uses top-1 for operation and query and
top-2 with equal forward weights for bindings. Its backward surrogate is the
corresponding softmax. Header scores never inspect packet contents.

The located query and operation states are `q` and `o`. Candidate and controls
reuse the frozen V23 comparator temperature and operation reader:

\[
m_j=\operatorname{softmax}_{j\in pair}
\left(\tau_{23}\langle q,A_j\rangle\right),
\]

\[
s=\sigma(U_{23}(o)),\quad
w_j=s m_j+(1-s)(1-m_j),
\]

\[
x_1=\sum_{j\in pair}w_jB_j^{image},
\qquad \hat y_1=C_{23}(x_1).
\]

Reread the actual generated pixels and bind them back to the visible source
glyphs:

\[
r_1=R(\hat y_1),
\quad b_j=\operatorname{softmax}_{j\in pair}
\left(\tau_{23}\langle r_1,B_j\rangle\right),
\]

\[
x_2=\sum_{j\in pair}b_jA_j^{image},
\qquad \hat y_2=C_{23}(x_2).
\]

All weighted image and state reductions are accumulated in FP64. The model has
no positional embedding, recurrent symbolic state, vocabulary projection, or
discrete glyph latent.

### Trainable Parameters

Every arm has exactly `1,347` trainable parameters:

- three role prototypes: `3 x 192 = 576`;
- four continuous null states: `4 x 192 = 768`; and
- three role-temperature scalars: `3`.

The null states are `header`, `query`, `operation`, and `history`. Prototypes
and nulls use seeded zero-mean Gaussian initialization with standard deviation
`0.02`.

### Matched Arms

- `packet_aware`: all visual causes available;
- `header_blind`: replace every header state with the normalized null-header
  state before routing;
- `query_blind`: replace the located query state with normalized null-query;
- `operation_blind`: replace the located operation state with normalized
  null-operation; and
- `history_blind`: replace only the reread frame-1 state with normalized
  null-history before frame 2.

All parameters exist in every arm with identical names and shapes.

## Training

Each batch jointly processes the original, query-counterfactual, and
operation-counterfactual prompts. Packet permutation is an exact structural
test, not an augmentation needed by the architecture. Distractor counts are
sampled per training episode.

For each of the two output frames, use the V23 topology loss and frozen-retina
generated-target cosine with weight `0.10`. Add a visual role-localization loss
with weight `0.25`: pixel L1 between routed content and four image-only targets
already present in the prompt (query image, operation image, mean binding-label
image, mean binding-glyph image). No role classification, target slot,
operation label, character label, or attention index enters a loss.

- AdamW, learning rate `2e-3`, betas `(0.9,0.95)`, weight decay `0.01`;
- 25-step linear warmup and cosine decay to `0.1` of base rate;
- gradient clip `1.0`, BF16, episode batch `64`;
- exactly `800` updates per arm, sequentially on one RTX 4090;
- validation every `100` updates on `512` deterministic development episodes;
- model seed `20260828`, dataset seed `20260829`; and
- smoke mode at most `20` updates and permanently non-evidentiary.

## Structural Tests Before Training

Tests must establish:

1. prompt lengths `15`, `18`, `21`, and `24` and output shape
   `[B,2,1,32,32]`;
2. packet-aligned dynamic padding uses only zero image tensors and supplies no
   mask or length to the model;
3. no integer or string tensor enters any `forward`;
4. exact `1,347` trainable parameters in every arm;
5. exact candidate/control parameter-name and shape equality;
6. retina, V23 operation reader, and canonicalizer are frozen and receive no
   parameter gradients;
7. role headers alone determine hard routing;
8. exactly two binding, one query, and one operation packet are selected;
9. packet permutation preserves both output frames within `1e-6`;
10. query-only change has output L1 `<1e-7` in `query_blind` once packet roles
    are fixed correctly;
11. operation-only change has output L1 `<1e-7` in `operation_blind` once
    packet roles are fixed correctly;
12. frame-1 override has frame-2 output L1 `<1e-7` in `history_blind`;
13. candidate frame 1 backpropagates to header, query, operation, binding-label,
    and binding-glyph images;
14. candidate frame 2 backpropagates through generated frame-1 pixels;
15. checkpoint round-trip preserves configuration and boundary receipts; and
16. development construction does not instantiate frozen images.

## Candidate Development Gates

Rank eligible checkpoints by the minimum of frame-1 query switch, frame-1
operation switch, and frame-2 history switch; then frame-1 identity top-1,
frame-2 label top-1, mean pixel F1, and earlier step. Every gate must pass on
`512` development episodes:

1. frame-1 binary choice accuracy `>0.95`;
2. frame-1 query switch accuracy `>0.90`;
3. frame-1 operation switch accuracy `>0.90`;
4. held-out-combination minimum switch accuracy `>0.85`;
5. frame-1 identity top-1 over all 88 identities and four views `>0.75`;
6. frame-1 pixel F1 `>0.68`;
7. frame-1 target cosine `>0.82`;
8. frame-2 label top-1 over all eight labels `>0.95`;
9. frame-2 pixel F1 `>0.58`;
10. frame-2 target cosine `>0.80`;
11. autonomous frame-2 identity agrees with teacher-forced frame 2 `>0.95`;
12. frame-2 identity is consistent with generated frame-1 identity `>0.92`;
13. history-override frame-2 label switch accuracy `>0.90`;
14. history-override frame-2 output pixel L1 `>0.10`;
15. query, operation, and binding-set header localization accuracy each
    `>0.99`;
16. packet-permutation identity consistency for both frames `>0.99`;
17. packet-permutation output pixel L1 for both frames `<1e-6`;
18. distractor-counterfactual identity consistency for both frames `>0.95`;
19. held-out `T=24` frame-1 identity and frame-2 label top-1 each `>0.90`;
20. clean image-only boundary; and
21. frozen images instantiated equals zero.

The frame-2 F1 threshold is fixed below the measured `0.60103` minimum of a
pre-protocol, non-evidentiary V23-canonicalizer diagnostic over the eight label
characters. It cannot be changed after training begins.

## Controls And Fresh Paired Gate

Each control selects first by its own exact intervention receipt, then role
localization, mean F1, and earlier step. Controls have no answer-quality
minimum. Only selected endpoints enter one fresh `1,024`-episode paired audit.
It must verify:

1. candidate query switch exceeds `query_blind` by `>0.40`;
2. candidate operation switch exceeds `operation_blind` by `>0.40`;
3. candidate frame-2 history switch exceeds `history_blind` by `>0.40`;
4. candidate minimum role localization exceeds `header_blind` by `>0.40`;
5. candidate frame-1 identity top-1 exceeds query-, operation-, and
   header-blind controls by `>0.30`;
6. candidate frame-2 label top-1 exceeds history- and header-blind controls by
   `>0.30`;
7. query-, operation-, and history-blind controls retain output-pixel L1
   `<1e-7` for their hidden intervention;
8. candidate retains every arm-specific gate;
9. every receipt, source hash, and inherited checkpoint hash matches; and
10. all arm parameter names and shapes are equal.

Endpoint substitution is forbidden. A smoke checkpoint is never eligible.

## Opaque Review And Frozen Policy

After automatic and paired gates pass, create a fixed 48-episode development
review showing the packet images and both generated answer frames without text
transcription or answer labels. An opaque visual reviewer chooses which source
glyph frame 1 depicts and which visible label frame 2 depicts. Required score
is at least `44/48` for each frame, including `11/12` held-out-length episodes.
This is an agent visual audit unless an actual human independently performs it;
it must not be called a human study.

Only after that gate passes may a separate evaluator instantiate frozen
identity images once. It reruns every fixed candidate gate on `1,024` frozen
episodes without model selection, threshold changes, or repeated attempts.

## Stop Rules

- Do not tune V24 after the first non-smoke run of any arm.
- Do not inspect development targets during implementation beyond fixed
  automated metrics and the authorized opaque review.
- Do not instantiate frozen images before all development gates pass.
- Do not promote a checkpoint when any required candidate or control gate
  fails.
- Preserve negative logs and refusal receipts.
- Do not describe packet routing, routed copying, canonicalization, or a
  two-frame synthetic stream as general language understanding.
