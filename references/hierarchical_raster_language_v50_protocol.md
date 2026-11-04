# Hierarchical Raster Language V50: Frozen Mechanism Protocol

Date frozen: 2026-11-04

Status: implementation complete; production evidence not yet opened

## Question

Can a compact model read an ordered stream of Chinese writing images and draw
the next writing image without reducing the deployed interface to character
IDs, Unicode, OCR, a glyph inventory, a visual codebook, or an external language
model?

V50 tests this mechanism only. It does not test broad question answering,
historical-form generation, or parity with a frontier text model.

## Why the Previous Writer Is Rejected

The end-to-end whole-cell flow calibration generated glyph-like texture but did
not use language context correctly. On 256 fixed development windows, ordered
history reached `0.00391` selected identity accuracy, shuffled history reached
`0.00781`, and the V43 baseline was `0.04297`. Ordered pixel F1 was `0.42040`,
below V43's `0.44507`. The flow route is therefore rejected rather than scaled.

Its failure is structurally plausible. A 32-by-32 target is a multimodal random
variable: many different next characters are compatible with one history. A
single whole-cell denoising objective can minimize loss by learning the common
glyph manifold while ignoring a comparatively weak context state.

## Design

V50 uses two causal scales.

1. A global reader processes completed 32-by-32 raster cells. It is initialized
   from the measured V42 reader, whose ordered next-image top-1 was `0.19971`,
   compared with `0.18359` under shuffled history and `0.12256` for the bigram
   control. V42's failed whole-cell sampler is removed.
2. A local writer first draws an 8-by-8 coarse image of the whole next cell,
   then divides the final cell into sixteen raw 8-by-8 detail patches and draws
   them in row-major order. Every local position is conditioned on the global
   visual state, the reader's continuous visual anchor, and previously drawn
   raster units from the same cell.

This is the smallest useful synthesis of two established observations:

- [MEGABYTE](https://arxiv.org/abs/2305.07185) shows that a large global model
  between patches and a small local model within patches can model raw,
  tokenization-free sequences efficiently; and
- [PIXAR](https://aclanthology.org/2024.findings-acl.874/) shows that visible
  text can be generated autoregressively in pixel space.

V50 differs from a flat patch model by retaining one global step per writing
cell while confining one coarse choice and sixteen detail steps to a small local
network. The coarse image forces identity to be chosen visually before local
stroke refinement. It
differs from V43 and the rejected joint-flow calibration by factorizing the
next-image distribution autoregressively instead of asking one denoiser to
choose and draw a whole glyph at once.

## Raster Probability Model

Let $h_t$ be the global visual state after visible cells $X_{\leq t}$, and let
$a_t$ be its continuous image anchor. Let $c_t\in\{0,1\}^{64}$ be a fixed
max-pooled 8-by-8 view of the whole next cell, and split the full-resolution
cell $Y_t$ into raster patches $y_{t,1:16}\in\{0,1\}^{64}$. The local model
factorizes

\[
p(Y_t\mid X_{\leq t})=
p(c_t\mid h_t,a_t)
\prod_{j=1}^{16}p(y_{t,j}\mid h_t,a_t,c_t,y_{t,<j}).
\]

Each patch distribution is a learned mixture of eight Bernoulli raster fields:

\[
p(y_{t,j}\mid c)=\sum_{k=1}^{8}\pi_k(c)
\prod_{r=1}^{64}\operatorname{Bernoulli}
\left(y_{t,j,r};\sigma(\ell_{k,r}(c))\right).
\]

This distribution is continuous in its conditioning state but samples binary
sensor values directly. The mixture components are not vocabulary entries,
glyph IDs, or a reusable codebook. They are recomputed distributions for each
context and patch position.

## Condition-Use Constraint

Ordinary teacher forcing can let the local writer copy detail patches and
ignore earlier language context. V50 therefore applies its condition-use margin
to the coarse whole-cell image, before any target detail is visible. It trains
that image under its correct state $c$ and a deranged state $\tilde c$. In bits
per pixel,

\[
\mathcal L_{cf}=\max\left(0,m-
\left[\operatorname{NLL}(Y\mid\tilde c)-
\operatorname{NLL}(Y\mid c)\right]\right),\qquad m=0.02.
\]

The full loss is

\[
\mathcal L=\mathcal L_{raster}
+0.75\mathcal L_{cf}
+0.05\mathcal L_{visual\text{-}NCE}
+0.025\mathcal L_{anchor}.
\]

The last two terms preserve the ordering signal already measured in V42.

## Fixed Training Configuration

| Quantity | Value |
| --- | ---: |
| Global reader | V42, failed generator removed |
| Local width / layers / heads | 256 / 4 / 8 |
| Raster patch | 8 by 8 binary pixels |
| Generated raster units per cell | 1 coarse + 16 detail |
| Conditional mixtures | 8 |
| Batch | 8 streams |
| Selected cells per stream | 8 |
| Updates | 5,000 |
| Frozen-reader warm start | 750 updates |
| Writer peak learning rate | `2e-4` |
| Reader peak learning rate | `1e-5` |
| Precision | BF16 |
| Device budget | one RTX 4090, less than 18 GiB allocated |

The corpus, partition, renderer, and 1,024-image evaluator bank remain those of
V42. The bank is evaluator-only and cannot enter model training or inference.

## Frozen Development Decision

V50 is retained only if all of the following hold on fixed development windows:

1. correct-context raster NLL is lower than deranged-context NLL on more than
   60% of examples, with a positive mean gain;
2. ordered first-sample visible identity exceeds V43's `0.04297`;
3. ordered identity exceeds shuffled-history identity by at least `0.01`;
4. ordered pixel F1 exceeds V43's `0.44507` without increasing blank output;
5. four-cell visible feedback remains finite and retains a measurable advantage
   over shuffled history; and
6. the deployed boundary contains only raster input, raster output, and learned
   continuous state.

If target likelihood improves but autonomous identity does not, the local
factorization is insufficient and will be rejected. A visually sharp gallery,
training loss, oracle best-of-many selection, or OCR non-emptiness cannot
replace these gates.

## Scale Boundary

A successful V50 would establish a working image-native language mechanism, not
general frontier-model equivalence. Broad knowledge requires substantially more
data and capacity. A later 0.3--1B-parameter model can fit a 24-GiB 4090 with
checkpointing and memory-efficient optimization, and an offline 4B/8B teacher
can prepare prompt--answer pages for domain distillation. The deployed model
must remain independent, and parity claims must be limited to measured tasks.
