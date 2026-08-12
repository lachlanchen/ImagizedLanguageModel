# V24 Research Decision: Visual Packets And Causal Rereading

Date: 2026-08-13

Status: architecture decision before V24 implementation, smoke training, or
evaluation

## Decision

V24 will test the first variable-length answer-image stream. It will not train a
page diffusion model, imitate OCR, or claim open-domain language. The bounded
question is:

> Can a compact student locate instructions by their rendered appearance in a
> variable-length image stream, execute the already-proven visual relation,
> emit one Chinese glyph image, reread its own pixels, and then emit the visible
> label bound to that glyph?

The prompt and answer are both tensors of images. Text is accepted at the outer
interface only by rendering it; strings, token IDs, Unicode IDs, OCR, role IDs,
and target indices never enter the learned path. The second answer frame is a
causal test rather than decoration: replacing the generated first frame with
the other source glyph must switch the second frame to the other visible label.

## Why This Is The Next Core Test

V23 passed a sealed same/other relation test over unseen Chinese identities, but
its six frame positions were architecturally named. It therefore established a
visual relation and a readable image actuator, not visual syntax or an
autoregressive image stream. Scaling that circuit directly to a page would
confound three unanswered questions:

1. can visible content, rather than absolute position, identify an instruction;
2. can the same mechanism ignore irrelevant visual packets and unseen lengths;
3. does a later answer actually depend on pixels emitted earlier?

V24 isolates those questions with two output frames. A positive result is the
smallest defensible bridge from one image answer to a text-image stream or
movie. A negative result identifies whether parsing, routing, or visual
feedback failed before expensive language pretraining begins.

## Relevant Primary Work

### Language directly in pixels

[PIXAR](https://arxiv.org/abs/2401.03321) demonstrates autoregressive language
generation in rendered pixel space and reports that a maximum-likelihood image
objective can produce noisy writing. It establishes that generation need not
end in a symbolic vocabulary, while motivating V24's separately qualified
glyph actuator and topology loss.

[PIXEL](https://arxiv.org/abs/2207.06991) shows that rendered text can support
language transfer without a tokenizer, but its masked-patch encoder is not an
autoregressive image writer. V24 keeps the perceptual input boundary and tests
generated visual feedback explicitly.

[Autoregressive Image Generation without Vector
Quantization](https://arxiv.org/abs/2406.11838) shows that continuous-valued
autoregression can use a diffusion loss instead of a VQ codebook. V24 does not
need diffusion for its routed glyph outputs, but preserves the same important
design freedom: an autoregressive unit can be continuous and image-valued.

### Content-addressed, exchangeable visual structure

[Slot Attention](https://arxiv.org/abs/2006.15055) produces exchangeable,
task-dependent visual slots through competitive attention and reports
generalization to unseen compositions. V24 uses a much smaller supervised
analogue: learned visual role prototypes compete over rendered packet headers,
and packet order is mathematically irrelevant.

[Deep Sets](https://arxiv.org/abs/1703.06114) and [Set
Transformer](https://arxiv.org/abs/1810.00825) motivate enforcing permutation
structure rather than hoping positional augmentation teaches it. V24 contains
no positional embedding and aggregates selected packets commutatively in
double precision.

[Perceiver IO](https://arxiv.org/abs/2107.14795) demonstrates a common
attention interface for arbitrary structured inputs and outputs with input
cost linear in stream size. V24 adopts the latent-query principle at much
smaller scale: three learned visual role queries address a variable number of
image packets.

### Relations and generated-history exposure

[Relation Networks](https://arxiv.org/abs/1706.01427) show the value of making
relational computation explicit when convolutional features alone do not solve
the task. V23 supplies that qualified relation primitive; V24 changes how its
operands are found and adds a second causal visual step.

[Scheduled Sampling](https://arxiv.org/abs/1506.03099) identifies the mismatch
between training on ground-truth history and inference on generated history.
V24 avoids that mismatch for frame 2: training and autonomous evaluation both
reread the model's generated frame 1. Teacher-forced frame 1 is only an
evaluator comparison.

## Visual Packet Grammar

Each packet is three consecutive `32x32` grayscale images:

```text
[visible header, visible content A, visible content B]
```

The fixed bounded vocabulary is:

| Header | Packet image sequence | Meaning in V24 |
|---|---|---|
| `契` | `[契, label, glyph]` | one visible binding |
| `法` | `[法, 同/异, blank]` | visible operation |
| `问` | `[问, label, blank]` | visible query |
| `旁` | `[旁, unrelated glyph, blank]` | distractor |
| `止` | `[止, blank, blank]` | visible end marker |

Every episode contains two binding packets, one operation packet, one query
packet, zero to three distractors, and one end packet. All packets except the
end marker are randomly permuted. Training sees zero, one, or two distractors;
development reserves three distractors as an unseen length. Batch padding is
made only of blank image packets, and no length or padding mask is passed to the
student.

This grammar still has fixed semantic roles. The advance over V23 is narrower
and measurable: roles no longer occupy fixed absolute frames and sequence
length is no longer fixed.

## Chosen Circuit

Write packet `j` as `(h_j,c_j,d_j)` and let the frozen visual retina emit
normalized states `R(.)`. Three learned normalized prototypes
`p_pair,p_op,p_query` score only visible header images:

\[
e_{jr}=\tau_r\langle R(h_j),p_r\rangle.
\]

Straight-through hard visual routing selects the highest-scoring query and
operation packets and the two highest-scoring binding packets. The forward
pass is discrete enough to prevent low-probability content leakage; its
backward pass uses softmax derivatives. No packet index, role label, or marker
identity enters the model.

The selected query and operation contents form continuous states `q` and `o`.
For each selected binding packet, the frozen V23 relation comparator computes

\[
m_j=\operatorname{softmax}_j
\left(\tau_m\langle q,R(c_j)\rangle\right),
\qquad s=\sigma(U_{23}(o)).
\]

For two bindings, `u=1-m` is the other-label distribution. The first routed
image and first generated frame are

\[
w=sm+(1-s)u,
\qquad x_1=\sum_jw_jd_j,
\qquad \hat y_1=C_{23}(x_1).
\]

The model then rereads its actual output pixels:

\[
r_1=R(\hat y_1),
\qquad
b_j=\operatorname{softmax}_j
\left(\tau_m\langle r_1,R(d_j)\rangle\right),
\]

\[
x_2=\sum_jb_jc_j,
\qquad \hat y_2=C_{23}(x_2).
\]

Thus frame 2 cannot inspect the original query or operation route. It can only
compare the reread first-frame image with visible source glyph images and route
their associated label image.

## Why Hard Visual Routing Is Appropriate Here

A soft pair mask gives every non-pair content a small path into the relation
normalizer. That can make a nominally query-blind control change when only the
query pixels change. Straight-through top-k selection gives the forward model
an auditable packet boundary while retaining gradients to the visual header
prototypes. It is a perceptual routing decision, not a language token: the
selected objects remain continuous images and retinal states, and the model
has no symbol table or output vocabulary.

The role prototypes are randomly initialized. They are trained using answer
images plus four visual localization targets: the selected query image, the
selected operation image, the mean of the two binding-label images, and the
mean of the two binding-glyph images. These are pixel tensors already visible
in the prompt; no discrete role classification loss is used.

## Causal Controls

All arms contain identical parameter names and shapes:

- `packet_aware`: normal visual packet stream;
- `header_blind`: replace only header retinal states with a learned continuous
  null before role routing;
- `query_blind`: replace only the located query state with a learned null;
- `operation_blind`: replace only the located operation state with a learned
  null; and
- `history_blind`: replace only the reread frame-1 state with a learned null
  before generating frame 2.

The decisive generated-history intervention replaces frame 1 at the reread
boundary with the canonical image of the other visible source glyph. The
packet-aware model must switch frame-2 label identity. The history-blind arm
must be invariant. This separates an autonomous visual continuation from a
second output independently decoded from the original prompt.

## Efficiency

The V16 retina, V23 operation reader, V23 relation temperature, and V23
canonicalizer are frozen. V24 trains only three `192`-dimensional role
prototypes, four matched `192`-dimensional null states, and three scalar role
temperatures: `1,347` parameters. The answer stream is produced in two causal
passes with memory linear in the number of packets. This is a mechanism test
that fits far below one RTX 4090's capacity; it is not yet a matched throughput
or quality comparison with an LLM.

## Alternatives Rejected For V24

- **A generic Transformer over all frames:** reintroduces absolute-position and
  attention-collapse shortcuts before content routing is proven.
- **OCR followed by an LLM:** solves a different problem through symbols and
  cannot generate unencoded forms natively.
- **A learned finite glyph codebook:** turns unseen-form writing into
  classification and cannot represent arbitrary marks.
- **Teacher forcing frame 2:** avoids the exact generated-history question V24
  is meant to answer.
- **Full-page diffusion now:** makes readability, syntax, factual knowledge,
  packet parsing, and visual feedback fail together and is not diagnostic on a
  single GPU.
- **Historical etymology answers now:** historical claims require provenance,
  retrieval controls, and copy-versus-synthesis labels after the stream
  mechanism works.

## Meaning Of A Positive Result

A passing V24 result would show a compact image-only model executing one
variable-position visual grammar and generating a two-frame causal image
stream over unseen Chinese identities. It would justify extending the same
interface to short rendered phrases and then page strips.

It would not prove arbitrary prompt understanding, book-scale language
modeling, factual etymology, free synthesis of oracle or bronze forms, page or
movie generation, human-like reading, Qwen parity, or greater efficiency than
token LLMs. Those remain empirical milestones, not assumptions.
