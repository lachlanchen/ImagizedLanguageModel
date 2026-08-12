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

## V19 Consequence

V19 added a spatial field as an optional residual to a complete frozen global
writer. The correct field beat a shuffled field by only `0.00879` dense F1,
against a fixed `0.12` requirement. This shows that ordinary reconstruction loss
allows the adapter to become a shared polish while the global route remains
sufficient.

The next model must change information routing, not only loss weights.

## Recommended V20: Retinal Topology Router

The smallest decisive experiment is a field-primary deterministic motor plan:

\[
p=\sigma\left(U(G(z,s))+(I-UD)H(F,s)\right).
\]

Here:

- `F` is the continuous local retinal field and drives spatial topology;
- `z` is the continuous global retinal state;
- `s` is image-derived style;
- `G` emits only a coarse plan;
- `D` and `U` are fixed down/up-sampling operators; and
- `(I-UD)H` is a high-frequency local component unavailable to the global path.

A stricter arm removes the global spatial seed and allows `z` only as
spatially-uniform FiLM modulation of a field-driven decoder. This tests whether
global semantics can guide local visual writing without redrawing the topology
from a class-like global vector.

### Fixed comparison

Train capacity-matched arms on one new salted split:

1. global-only coarse planner;
2. field-primary planner;
3. field-primary plus global channel modulation; and
4. the V19 additive-residual design as a negative control.

For each arm report correct field, shuffled field, zero field, local occlusion,
and shuffled global state. Preserve the V19 complexity strata and blinded
recognition rubric. Predeclare gates before training, including a substantial
dense field-causality margin, identity, readability, FLOPs, peak VRAM, and
throughput.

### What not to do next

- Do not tune V19 thresholds after its failed audit.
- Do not increase model width to hide a routing failure.
- Do not couple the writer to the weak language core before isolated actuation
  passes.
- Do not use OCR/text embeddings in the student to obtain semantic alignment.
- Do not train a page-scale diffusion model before a 32-cell closed loop remains
  readable.

## Roadmap Placement

V20 is still P1.5, not instruction following. After a field-causal writer passes
development, blinded recognition, and a new frozen evaluation, the order is:

1. predict local and global future visual state from causal image history;
2. run the accepted writer on predicted rather than supplied state;
3. reread the generated pixels for 32 regions without losing readability;
4. use uncertainty-triggered compute only if it improves the matched curve; and
5. begin the bounded Visual Word-Origin Book curriculum from open or
   rights-cleared image trajectories.

This keeps the broader model innovative but the next experiment small enough to
fail clearly on one RTX 4090.
