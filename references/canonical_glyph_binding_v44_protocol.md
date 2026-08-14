# Canonical Glyph Binding V44 Protocol

Status: preregistered before V44 implementation or training.

Date: 2026-08-14

## Question

V42 is the accepted base mechanism. Its `24,346,497`-parameter image-only
causal reader uses ordered `64 x 1 x 32 x 32` Chinese glyph rasters and beats
image-unigram, symbolic-bigram, and shuffled-history controls on a fixed
development audit. It does not reliably bind earlier history to the correct
next image: exact-suffix arm accuracy is `0.53027`.

V43 fine-tuned the whole reader on 5,000 same-suffix pairs. A post-result audit
found `0.99219` arm accuracy on sampled training pairs but only `0.58301` on
unseen training pairs and `0.54688` on development. The same V43 spatial writer
reaches `0.88240` pixel F1 when supplied an evaluator-exact ink plan, compared
with `0.46110` from the autonomous plan. V44 therefore changes only the reader
test. It asks whether a small, explicitly long-history residual can generalize
counterfactual image binding without damaging the accepted V42 language path.

## Claim Boundary

A passing V44 may claim that a frozen image-native causal reader can acquire a
generalizable long-history correction from raster-only contexts and raster
targets on one consumer GPU. It may reopen the already measured V43 writer for
a later composition test. It may not claim readable autonomous generation,
question answering, historical-form generation, parity with a text LLM,
compute superiority, or a complete ILM.

## Fixed Data Boundary

- Corpus: `data/visual_grammar/chinese_wikisource_public_domain.jsonl`.
- Required manifest SHA-256: the frozen V25 manifest receipt.
- Base reader: V42 checkpoint SHA-256
  `a5ce2ff20d0fc6d336f489b1e2d29bb96a1d0666399c2e2b95e7ac33a9abe870`.
- Training uses only the existing training partition.
- Selection uses only the existing development partition.
- The frozen partition remains sealed.
- Rendering remains Noto Sans CJK Regular at 26 pixels in a `32 x 32` cell,
  without font or geometric augmentation.
- Original and Simplified strings are host-side corpus preparation only.
- Learned modules receive floating-point images and image-derived continuous
  states only.

Host code deterministically finds unique context pairs with the same final
four glyph images and different next glyph images. The first 24,000 pairs are
consumed exactly once. The next 1,024 unique suffixes form a training-partition
holdout and are never passed to the optimizer. Strings, identifiers, suffixes,
and character values are removed before collation. Candidate columns are
deterministically permuted.

## Frozen Base And Tangent Residual

Let the frozen V42 reader be `B`. For a 64-image context `X`, it emits causal
hidden states `H`, terminal anchor `a_0`, and fixed DCT image fields
`phi(X)`. A second frozen pass over the final four images emits suffix state
`h_s`. V44 excludes those final four images from its memory keys and forms

\[
P = \operatorname{LN}(W_h H_{1:60} + W_x\phi(X_{1:60}) + E_{rel}),
\]

\[
q = W_q[h_{64};h_s;h_{64}-h_s].
\]

Two six-head cross-attention blocks with width 192 let `q` read `P`. Their
output predicts a full 1,024-dimensional residual `r`. Only the component
tangent to the base unit field may change the plan:

\[
r_\perp = r-(r^\top a_0)a_0,\qquad
a = \operatorname{norm}(a_0 + 0.5r_\perp).
\]

The final residual projection is initialized to zero, so V44 is exactly V42 at
update zero. Every V42 parameter, including its generator and contrastive
temperature, remains frozen. Contexts of four images or fewer bypass the
residual, making the suffix-four and last-image controls exact base controls.
The adapter must contain fewer than 2.0 million trainable parameters.

## Raster-Only Binding Objective

For a paired example, contexts `X_0,X_1` share their final four images. Their
permuted candidate image fields are `c_0,c_1`, and `pi(i)` gives the correct
image column for arm `i`. The model scores

\[
S_{ij}=s\,a_i^\top c_j.
\]

`L_pair` is row-wise cross entropy. The signed pair displacement must also
match the signed target-image displacement:

\[
L_{delta}=1-\cos(a_0-a_1,c_{\pi(0)}-c_{\pi(1)}),
\]

with a Huber penalty `L_norm` on the two displacement norms. `L_pair-anchor`
is mean target-anchor cosine distance. A deterministic image-only permutation
of positions 1 through 60 leaves the final four images fixed; `L_order`
requires the ordered context to score its target at least `0.05` above this
shuffled-prefix version.

Each update also receives ordinary natural raster windows. `L_natural` is
dynamic image contrast over their terminal targets, `L_target` is terminal
target-anchor cosine distance, and `L_distill` is cosine distance to the frozen
V42 terminal anchor. The fixed objective is

\[
\begin{aligned}
L_{V44}={}&2.0L_{pair}+1.0L_{delta}+0.25L_{norm}
 +0.5L_{pair-anchor}+0.5L_{order}\\
&+0.75L_{natural}+0.25L_{target}+0.5L_{distill}.
\end{aligned}
\]

No candidate image bank, text model, OCR output, token ID, Unicode value,
character ID, or lookup result enters the adapter or deployed runtime.

## Fixed Optimization

| Setting | Value |
|---|---:|
| updates | 3,000 |
| ordinary stream batch | 8 |
| pair batch | 8 pairs |
| unique train pairs | 24,000 |
| unseen train-partition pairs | 1,024 |
| pair suffix | 4 glyph images |
| residual width | 192 |
| residual layers | 2 |
| residual heads | 6 |
| residual dropout | `0.10` |
| residual scale | `0.50` |
| peak learning rate | `2e-4` |
| warm-up | 150 updates |
| final learning-rate ratio | `0.10` |
| AdamW betas | `(0.9, 0.95)` |
| weight decay | `0.05` |
| gradient clip | `1.0` |
| training precision | BF16 on CUDA |

## Fixed Development Audit

The evaluator runs the frozen V42 base and V44 on the same examples:

- 2,048 fixed natural-language development windows;
- a 1,024-image evaluator-only character bank;
- 512 fixed development exact-suffix pairs;
- 1,024 sampled consumed training pairs, for diagnosis only; and
- the 1,024 preregistered unseen training-partition pairs.

Natural-language variants remain full, suffix-four, last-image,
suffix-preserving shuffled history, blank history, image unigram, and symbolic
bigram. Pair variants remain full, suffix-four, last-image, and shuffled
earlier history. No writer or frozen example is opened in V44.

## Conjunctive Gates

V44 passes only if every condition is true:

1. full natural top-1 exceeds image unigram by more than `0.03`;
2. full natural top-1 exceeds symbolic bigram by more than `0.01`;
3. ordered target log probability exceeds shuffled history by more than
   `0.05` nat;
4. full natural top-1 exceeds shuffled history by more than `0.015`;
5. V44 full natural top-1 is no more than `0.015` below the matched frozen V42
   base;
6. V44 full target log probability is no more than `0.10` nat below the
   matched frozen V42 base;
7. development exact-suffix full arm accuracy exceeds `0.60`;
8. development full arm accuracy improves over matched V42 by more than
   `0.05`;
9. development full arm accuracy exceeds shuffled-prefix arm accuracy by more
   than `0.04`;
10. unseen training-partition full arm accuracy exceeds `0.60`;
11. consumed-pair minus unseen-pair arm accuracy is below `0.10`;
12. the image-only runtime boundary is clean;
13. the V42 base is byte-for-byte frozen and the residual has fewer than
    2.0 million trainable parameters; and
14. peak allocated CUDA memory is below 18 GiB.

Thresholds, seeds, pair ordering, loss weights, and architecture dimensions
may not change after a production development result is observed. A smoke run
may test integration but cannot support a capability claim. If any gate fails,
V44 is recorded as rejected or partial, the V43 writer remains closed, and the
frozen partition remains sealed.
