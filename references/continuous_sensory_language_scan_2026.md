# Continuous Sensory Language: 2026 Decision Scan

Date: 2026-08-12

Purpose: identify current methods that can strengthen the next ILM experiment
without weakening the strict pixels-to-continuous-state-to-pixels student
boundary.

## Boundary Used For This Scan

Useful inspiration is not automatically an admissible ILM component. The
deployed student may receive writing pixels and continuous visual states. It may
not receive token IDs, bytes, Unicode IDs, OCR text, a discrete visual codebook,
a candidate glyph table, or a hidden pretrained text model. Such systems remain
important baselines and offline teachers.

## Primary Sources And Decisions

| Work | Primary result | ILM decision |
|---|---|---|
| [Byte Latent Transformer](https://arxiv.org/abs/2412.09871) | Dynamically groups bytes into entropy-triggered patches and reports better fixed-FLOP scaling than fixed byte processing. | Borrow adaptive-compute logic, not bytes or patch tokens. Trigger extra visual computation where retinal prediction uncertainty is high. |
| [Scratchpad Patching](https://arxiv.org/abs/2605.09630) | Adds transient within-patch updates to reduce stale context and reports smaller KV/cache and compute at long byte patches. | Test transient local visual scratch state inside a fixation or line. Keep it continuous and compare against a fixed-stride control. |
| [Continuous Visual Autoregressive Generation via Score Maximization](https://arxiv.org/abs/2505.07812) | Uses strictly proper scores, especially the energy score, for autoregression over continuous visual values without vector quantization. | Strong candidate for calibrated continuous next-field uncertainty; compare it with cosine/NCE instead of treating raw energy margins as probability. |
| [VL-JEPA](https://arxiv.org/abs/2512.10942) | Predicts continuous target-text embeddings and selectively invokes a text decoder. | Supports continuous predictive state and selective decoding, but its text encoder/decoder violates the independent ILM boundary. Use only as a benchmark and conceptual control. |
| [TextLDM](https://arxiv.org/abs/2605.07748) | Applies latent diffusion to language and reports that reconstruction latents are insufficient without alignment to a frozen language model. | Important warning: image reconstruction alone does not create semantics. The frozen text-model alignment route is forbidden for the final student; replace it with causal visual prediction and image-only cross-view/trajectory alignment. |
| [CLEAR](https://arxiv.org/abs/2508.19098) | Models compact continuous speech latents with a lightweight per-state rectified-flow head. | Evidence that continuous autoregression plus a small flow head is practical. Reserve for a later speech extension; it does not solve visual writing topology. |
| [Sparse Image Synthesis via Joint Latent and RoI Flow](https://proceedings.neurips.cc/paper_files/paper/2025/hash/d6f7e8076b5da40e2c55582b3707419e-Abstract-Conference.html) | Generates sparse positional RoI latents and reports competitive ImageNet synthesis with 64 latents. | Later answer-page layout can allocate compute to ink regions instead of paper background. Do not introduce this complexity before the isolated writer passes. |
| [Align Your Flow](https://arxiv.org/abs/2506.14603) | Distills continuous-time flow maps for effective few-step generation. | Candidate only after topology is stable. Few-step surface rendering cannot repair a language or routing failure. |

## Mathematical Opportunity

### Proper continuous prediction

For a conditional predictive distribution `P` over a future visual state and
observed state `y`, a sample energy score is

\[
S(P,y)=\mathbb E\|X-y\|_2-
\tfrac12\mathbb E\|X-X'\|_2,
\qquad X,X'\sim P.
\]

The first term rewards accuracy and the second prevents collapse by rewarding
appropriate sample spread. Unlike an unnormalized raw energy margin, a strictly
proper score has a distributional target. A bounded future experiment should
compare this objective with the current proposal/NCE and spherical-flow losses
under matched parameters and FLOPs.

### Uncertainty-triggered visual compute

Let `u_t` be calibrated uncertainty for the next retinal region. A visual
scratch update is permitted only when `u_t` crosses a fixed threshold:

\[
k_t=k_{\min}+\sum_j \mathbf 1[u_t>\tau_j].
\]

The model then spends extra local recurrent or attention updates on dense,
damaged, ambiguous, or historically unfamiliar writing. Acceptance requires a
quality-versus-FLOPs curve against a fixed number of updates; entropy gating is
not accepted merely because it sounds biologically plausible.

## V19 to V20: Routing Decision

V19 added a spatial field as an optional residual to a complete frozen global
writer. The correct field beat a shuffled field by only `0.00879` dense F1,
against a fixed `0.12` requirement. Ordinary reconstruction therefore learned
a shared polish while the global route remained sufficient.

V20 changed the causal graph. Its global branch emitted only coarse `4x4` block
logits; the continuous local field emitted zero-mean `8x8` detail inside each
block. An exact-capacity control sent repeated global state through the same
local decoder. Both arms had `506,448` trainable parameters and used the same
salted split, compute budget, and frozen V16 retina.

The structural intervention worked. Candidate dense F1 was `0.70131`, versus
`0.57953` under a shuffled field and `0.34003` under a zero field. The gains
`+0.12177` and `+0.36128` pass the fixed causal margins. Occluding one field
quadrant changed only the corresponding output quadrant (`1.0` locality), while
all field interventions had exactly zero effect in the global-repeat control.

The writer did not pass. Candidate overall F1 (`0.63608`) and target cosine
(`0.81519`) missed their fixed thresholds, as did the strict floating-point
detail-mean invariant (`2.03e-6` versus `<1e-6`). Its final dense advantage over
the equal-capacity control was only `+0.01791`, below the fixed paired target of
`>0.03`. Neither arm selected, so no paired audit, human review, or frozen query
was authorized.

The decision is narrow and useful: topology-necessary local routing is now
demonstrated, but reserving only high-frequency residual detail for the field is
not a sufficient visual writer.

## V21 Result: Field-Complete Routing Works, Disjoint Patches Do Not

V21 implemented the smallest decisive field-complete model. The topographic
field determines both coarse occupancy and fine detail:

\[
(\gamma,\beta)=M(z,s),\qquad
c=C_F(\gamma\odot F+\beta),\qquad
d=B_0A_F(\gamma\odot F+\beta),\qquad
p=\sigma(U(c)+d).
\]

Here:

- `F` is the continuous local retinal field;
- `z` is continuous global state and `s` is image-derived style;
- `M` emits only channelwise modulation shared over spatial locations;
- `C_F` derives coarse occupancy from the field;
- `A_F` derives local detail through a bounded receptive field; and
- `B_0` is a fixed zero-mean basis, making the decomposition exact by
  construction rather than numerical recentering.

The exact-capacity control replaces `F` with global state tiled over the same
grid. Both arms have `582,336` parameters. At candidate step 1,400, dense F1 is
`0.70535`, versus `0.53145` for shuffled `F` and `0.35880` for zero `F`.
The fixed gains `+0.17390` and `+0.34654` pass; identity top-1 is `79.102%`,
target cosine is `0.83313`, and quadrant locality is `1.0`. DC leakage, basis
Gram error, zero-source cell variation, and decomposition error are exactly
zero in the recorded snapshot. The tiled-global control produces repeated
patch textures and reaches only `0.14733` overall F1 at its selected structural
step.

The candidate does not select. Overall F1 (`0.60378`), simple F1 (`0.56483`),
and medium F1 (`0.59449`) miss the fixed `0.66`, `0.58`, and `0.60` gates. No
formal paired audit, human review, or frozen query is authorized.

The causal question is answered positively and the writer question negatively:
a local continuous field can carry the complete spatial plan, but independent
nonoverlapping `8x8` patch emission introduces seams and weak thin-stroke
continuity. The next writer should use overlapping local support with a fixed
partition-of-unity blend, or a matched bounded multiscale local operator. Its
influence radius must remain measurable so continuity does not reintroduce a
global character unembedding.

## Visual Language Stream Contract

The long-term input/output standard can cover flat books and 3D writing without
changing the representational principle:

\[
X\in[0,1]^{T\times D\times H\times W\times C}.
\]

`T` is visual sequence or time, `D` is optional geometric depth, `H,W` are
space, and `C` is the sensor channel. A page has `T=1,D=1`; an ordered book or
line stream has `T>1,D=1`; a 3D glyph string has `D>1`; and a 3D character movie
uses both depth and time. Chinese oracle, bronze, seal, traditional, simplified,
Kanji, Latin, and other writing are visual observations in the same stream, not
entries in separate deployed vocabularies.

For prompt following, the same contract is partitioned into observed prompt and
generated answer frames:

\[
X_{\mathrm{prompt}}\rightarrow
Y_{\mathrm{answer}},\qquad
Y_{\mathrm{answer}}\in
[0,1]^{T_a\times D\times H\times W\times C}.
\]

Typed text is rendered at the boundary and photographed writing enters
natively. `T_a=1` is an answer image; `T_a>1` is a text-image stream or movie.
A valid language test requires held-out prompt-dependent answer changes, not
only reconstruction or glyph identity.

This unification is a goal, not present evidence. V21 remains 2D and is an
isolated writer supplied with intended state. Depth and motion are added only
after a small `D=1` model reads a prompt, predicts answer state, writes, and
rereads successfully.

### What not to do next

- Do not tune V21 thresholds or reinterpret its best diagnostic snapshot as a
  selected writer.
- Do not increase model width to hide a routing or objective failure.
- Do not couple the writer to the weak language core before isolated actuation
  passes.
- Do not use OCR/text embeddings in the deployed student to obtain semantic
  alignment.
- Do not train page-scale, 3D, or video diffusion before a 32-region 2D loop
  remains readable.

## Roadmap Placement

V21 remains a P1.5 causal-routing result, not a selected writer or instruction
following. The ordered route is now:

1. preregister a continuity-preserving local writer and exact-capacity
   tiled-global control;
2. pass automatic selection, numeric blinded recognition, and one new frozen
   writer evaluation;
3. train an image-only prompt-conditioned model to predict local and global
   answer fields, with prompt-shuffle, last-frame, unigram, symbolic bigram, and
   held-out counterfactual controls;
4. run only an accepted writer on predicted fields and reread a fixed
   multi-frame answer stream;
5. begin the bounded Visual Word-Origin Book curriculum from open or
   rights-cleared 2D prompt/answer trajectories; and
6. extend the same Visual Language Stream to page scale, geometric depth, and
   character movies only after the 2D prompt-to-answer loop is accepted.

This keeps the broad interface genuinely new while making the next experiment
small enough to fail clearly on one RTX 4090.
