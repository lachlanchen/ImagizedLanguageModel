# Exact raster density V54: frozen experiment

Date: 2026-11-04. Frozen before implementation, training or development scoring.

## Question and scope

Can an exact conditional distribution over raw writing pixels preserve an
explicit image-copying path and turn the measured V42 visual state into useful
next-image samples? This is a language-mechanism experiment on the full public
Chinese corpus, not the final general instruction-following ILM.

V53's target-supplied writer only reached 0.459 pixel F1, whereas the fixed
inverse transform copied exactly. V54 therefore uses that inverse as its
aligned pixel condition. It learns a normalized raster probability law rather
than another unconstrained image redraw, score-combination or pixel marginal.

This applies the conditional raw-pixel factorization established by
[PixelRNN](https://proceedings.mlr.press/v48/oord16.html) and
[conditional PixelCNN](https://arxiv.org/abs/1606.05328), using a
[MADE](https://arxiv.org/abs/1502.03509) density within each row. These components
are prior work. The experiment tests their integration with this repository's
image-only language state and exact raster coordinate on one RTX 4090.

## Distribution and strict causality

For a binary next raster X, visible history H and rows r=0..31, columns c=0..31:

    p(X | H) = product_r product_c Bernoulli(X[r,c]; sigmoid(logit[r,c]))
    logit[r,c] = 6 * plan[r,c] + delta_theta(H, X[<r,:], X[r,<c])

The plan is the fixed inverse of V42's normalized image field multiplied by
32, with signed values clamped to [-2,2]. For a clean supplied image field,
this plan is exactly +/-1. The learned residual is zero at initialization,
so the mode copies any supplied binary image exactly, including unseen forms.
That copying property is architectural and must not be described as learned
language. For an uncertain predicted plan, the normalized autoregressive
distribution may correct globally inconsistent stroke combinations.

The row transformer reads shifted observed rows and the continuous condition.
A masked MLP emits each pixel's Bernoulli logit using only earlier pixels in
that row. All 1,024 training probabilities can be evaluated together. Sampling
uses 32 transformer evaluations and 1,024 small masked-head evaluations per
glyph. Greedy thresholding is reported separately from the primary sample.

The objective is unweighted binary cross entropy summed into exact image NLL.
There is no foreground weighting, image-space MSE, Dice term, contrastive
margin, adversarial objective or learned energy selector. Such terms would
obscure whether the normalized conditional density itself learns language.

## Runtime boundary

Input and output are floating/binary raster tensors. The student has no text
tokenizer, character or Unicode identifiers, glyph vocabulary, codebook,
quantizer, OCR transcript, persistent image bank or external runtime teacher.
Pixel values and spatial coordinates are observed image measurements, not
linguistic IDs. Strings may prepare data offline. Evaluator banks may only be
created after generated images exist.

## Fixed components

- Reader: frozen V42, 16,277,249 parameters, checkpoint
  `artifacts/canonical_glyph_language_v42_20260814/checkpoint_final.pt`, SHA-256
  `a5ce2ff20d0fc6d336f489b1e2d29bb96a1d0666399c2e2b95e7ac33a9abe870`.
- Corpus: `data/visual_grammar/chinese_wikisource_public_domain.jsonl`, SHA-256
  `76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03`.
- Canonical V42 renderer, 32-pixel cells, fixed 26-pixel Noto Sans CJK font.
- 64 visible cells, original and simplified script views, same source splits.
- Row transformer: width 192, 4 causal layers, 6 heads, MLP ratio 3, no dropout.
- Within-row MADE: two hidden layers of width 256, SiLU, fixed natural pixel
  ordering; condition projections have no target-pixel dependence.
- Aligned plan projection and shifted row projection: each 32 pixels to 192.
- Global condition: concatenated V42 hidden384 and image field1024 to 192.
- Fixed learned row positions: 32 by 192; these index space, not written forms.
- New trainable density parameters at most 5M; combined runtime at most 23M.

## Fixed training

8,000 AdamW updates, learning rate 2e-4, betas (.9,.95), weight decay .01,
400-update warmup, cosine decay to .1 peak, norm clip 1, EMA .999, BF16,
seed 20265410, dataset seed 20265411, only CUDA physical device 0, memory <18 GiB.

Each update uses 8 public streams and 16 selected causal positions per stream,
four from each quarter including the final position. The frozen causal reader
computes all selected history states without seeing their future targets.
For 25% of rows, an explicit motor exercise supplies the true target field and
zeros the language hidden state. Remaining rows use the predicted visual
condition. Only the image likelihood trains the density head. The reader is
never updated. Supplied-image rows are labelled as motor exercises in receipts.

Checkpoints retain EMA, optimizer, source/protocol/foundation/corpus hashes,
data offset and all RNG states. Interrupted and continuous smoke training must
match exactly before production. Development is not opened during training.

## Locked development measurement

- Seed 20265420; 512 deterministic development windows, next-cell and 4-cell
  continuation tensors from the existing record partition.
- Exact NLL on all 512 real target images under ordered, suffix-preserving
  shuffled and zero conditions; paired record-bootstrap intervals (2,000 draws).
- First-sample and greedy outputs on 256 paired-noise prefixes, with no
  reranking, rejection, image lookup or OCR repair.
- 64 prefixes for four generated-image feedback steps.
- Supplied-image motor test on 256 targets, separately labelled, with exact
  inverse control and paired random-plan control.
- Pixel F1, blank rate, isolated ink, ink ratio, exact image match and nearest
  evaluator identity; no bank is deployed or used to select pixels.
- The evaluator includes all distinct target rasters when constructing its
  post-generation identity bank, including punctuation. Identity is still a
  nearest-raster diagnostic, never proof that a pseudo-glyph is human-readable.
- Fixed unselected galleries and per-example metrics retained.

## Predeclared decision

Continue toward broader language training only if all hold:

1. strict pixel causality and image-only boundary pass; no frozen/hash drift;
2. supplied-image greedy F1 >=.995 and sample F1 >=.98, and correct supplied
   image beats a deranged image condition by >=.30 F1;
3. ordered target NLL is lower than shuffled, with the record-bootstrap 95%
   interval for ordered advantage strictly above zero;
4. ordered first-sample F1 >=.48 and identity >=.03;
5. ordered first-sample identity exceeds shuffled by >.01 and has a positive
   record-bootstrap lower confidence bound;
6. sample isolated-ink fraction <=.01, blank rate <.02, ink ratio [.7,1.5];
7. ordered four-step feedback mean identity exceeds shuffled;
8. finite tensors, exact-resume smoke, declared resource/parameter limits pass.

The exact likelihood and motor tests may succeed while the full mechanism
fails. Report those distinctions without promoting copying, teacher forcing,
or image-bank assignment to general language capability. No later change to
ordering, temperature, thresholds, training budget or sampling count belongs
to this frozen run.
