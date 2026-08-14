# Canonical Glyph Flow V43 Protocol

Status: preregistered before V43 implementation or training.

Date: 2026-08-14

## Question

V42 established a bounded image-native language result on 2,048 development
windows. A `24,346,497`-parameter causal model received only ordered
`64 x 1 x 32 x 32` glyph rasters and reached `19.9707%` exact next-image
retrieval, above image unigram (`1.4160%`), symbolic bigram (`12.2559%`),
last-cell (`8.0566%`), suffix-four (`15.1855%`), and shuffled-history
(`18.3594%`) controls. It also improved ordered target log probability over
shuffled history by `0.17391` nat.

V42 is not a complete image language model. Its exact-suffix counterfactual
arm accuracy is only `53.0273%`, and its bank-free generated pixel F1 is only
`0.37308`. A deterministic inverse of the V42 anchor improves F1 to `0.47328`
but produces dense visual noise. The qualified V41 MX-Font motor does not
repair autonomous V42 anchors (`0.45530` F1 after the motor), so that bridge is
rejected for this purpose.

V43 asks two isolated questions:

1. Can train-only same-suffix image-pair supervision make the V42 reader bind
   longer visual history to the correct next image?
2. Can a small conditional spatial flow turn the accepted visual language
   state into a sharp `32 x 32` glyph image without a vocabulary, codebook,
   OCR system, glyph lookup, or deployed candidate bank?

## Claim Boundary

A passing V43 may claim a compact model learns bounded next-glyph dependencies
from Chinese raster streams and emits a readable next-glyph raster through a
fully image-only runtime. It may not claim general question answering, Qwen
parity, arbitrary historical-form generation, human-equivalent reading,
superiority to token language models, or a completed 1B-scale ILM.

## Fixed Data Boundary

- Corpus: `data/visual_grammar/chinese_wikisource_public_domain.jsonl`.
- Required manifest SHA-256: the frozen V25 manifest receipt.
- Reader initialization: the completed V42 checkpoint with SHA-256
  `a5ce2ff20d0fc6d336f489b1e2d29bb96a1d0666399c2e2b95e7ac33a9abe870`.
- Training uses only the existing training partition.
- Selection uses only the existing development partition.
- The frozen partition remains sealed.
- Canonical rendering remains Noto Sans CJK Regular at 26 pixels in a
  `32 x 32` cell, with no font or geometric augmentation.
- Original and Simplified views remain host-side corpus preparation only.
- Student tensors contain only floating-point images and image-derived states.

Text strings may be used offline to locate training windows with an identical
four-glyph suffix and different next glyphs. They are discarded before
collation. The reader receives two raster histories and two raster candidates;
candidate columns are deterministically permuted per example. No string,
character ID, Unicode value, or pair metadata reaches a learned module.

## Stage A: Long-History Binding

The V42 reader is initialized from the pinned checkpoint. Its obsolete V42
one-step generator is frozen. Each update combines:

1. one ordinary batch of eight 65-cell raster streams, preserving dynamic
   image contrast, continuous anchor alignment, and decoded-pixel objectives;
2. one batch of eight train-only context pairs with the same final four raster
   cells but different earlier history and different next-cell images.

For pair `b`, reader anchors `a_b0,a_b1` and permuted candidate image fields
`c_b0,c_b1` produce a `2 x 2` score matrix

\[
S_{bij}=s\,a_{bi}^{\mathsf T}c_{bj}.
\]

The paired loss is row-wise cross entropy against the image-column
permutation. It cannot be solved from the shared suffix or a fixed candidate
position. The fixed Stage-A objective is

\[
L_A=L_{dynamic}+0.25L_{anchor}+0.20L_{pixel}+2.0L_{pair}.
\]

Stage-A settings are fixed before training:

| Setting | Value |
|---|---:|
| updates | 3,000 |
| ordinary stream batch | 8 |
| paired-history batch | 8 pairs |
| pair suffix | 4 glyph images |
| train pair pool | 5,000 unique suffixes |
| peak learning rate | `5e-5` |
| warm-up | 100 updates |
| final learning-rate ratio | `0.10` |
| AdamW betas | `(0.9, 0.95)` |
| weight decay | `0.05` |
| gradient clip | `1.0` |

## Stage B: Conditional Spatial Flow Writer

The selected Stage-A reader is frozen. Its final hidden state `h`, normalized
continuous anchor `a`, and fixed inverse-DCT ink plan `p(a)` condition a
convolutional U-Net flow writer. For signed target pixels `x` and Gaussian
noise `epsilon`, sample `tau` uniformly and define

\[
y_\tau=(1-\tau)x+\tau\epsilon,\qquad u^*=\epsilon-x.
\]

The writer predicts `u_theta(y_tau,tau,h,p(a))`. It is trained with
stroke-weighted velocity MSE plus a `0.10` weighted endpoint L1 term. Ten
percent conditional dropout permits classifier-free guidance but supplies no
symbolic information.

Stage-B settings are fixed before training:

| Setting | Value |
|---|---:|
| writer updates | 5,000 |
| stream batch | 8 |
| sampled positions per stream | 16 |
| effective glyph images per update | 128 |
| U-Net base channels | 64 |
| condition width | 384 |
| flow context width | 256 |
| peak learning rate | `2e-4` |
| warm-up | 200 updates |
| final learning-rate ratio | `0.10` |
| condition dropout | `0.10` |
| endpoint weight | `0.10` |
| stroke weight | `2.0` |
| inference integration | 16 Heun steps |
| generated candidates | 4 continuous flows |
| guidance scale | `1.25` |

At inference, four pixel fields are integrated from independent Gaussian
noise. The frozen reader re-encodes those generated pixels, and its own anchor
selects the highest-cosine field. This selection has no candidate glyph bank.
The selected pixels are the model output and are the only feedback for another
generation step.

## Development Audit

The V42 development protocol remains unchanged:

- 2,048 fixed natural-language windows;
- a 1,024-image evaluator-only character bank;
- 512 exact-suffix counterfactual pairs;
- 256 bank-free generated next glyphs;
- full, suffix-four, last-cell, shuffled-history, blank-history, image-unigram,
  and symbolic-bigram controls.

The evaluator-only bank may identify an emitted image after generation. It is
never visible while producing or selecting the image.

## Unchanged Conjunctive Gates

V43 passes only if every condition is true:

- full top-1 minus unigram top-1 is greater than `0.03`;
- full top-1 minus bigram top-1 is greater than `0.01`;
- ordered target log probability minus shuffled is greater than `0.05` nat;
- full top-1 minus shuffled top-1 is greater than `0.015`;
- exact-suffix counterfactual arm accuracy is greater than `0.60`;
- generated identity top-1 exceeds unigram top-1;
- generated pixel F1 is greater than `0.55`;
- generated blank rate is below `0.02`;
- the image-only runtime boundary is clean; and
- peak allocated CUDA memory is below 18 GiB.

No threshold may be changed after observing V43 development results. A smoke
run may exercise integration but cannot support a capability claim. If either
Stage A or Stage B fails, V43 is recorded as a failed or partial mechanism and
the frozen partition remains sealed.

