# V42--V47 Predictive-State Diagnostic Plan

Date frozen: 2026-08-15

Status: post-result diagnostic plan, frozen before implementation or execution

## Purpose

V42 learned bounded next-glyph language in a direct canonical image field,
whereas V47 replaced that field with the qualified V34 visual codec sphere and
lost most held-out language performance.  This diagnostic asks whether the
loss is primarily explained by:

1. failure to generalize from the training-record partition;
2. destructive use of long visual history;
3. unsuitable geometry of the codec reconstruction state; or
4. collapse onto a small set of frequent visual predictions.

The diagnostic does not alter the V42 or V47 gate decisions, train either
model, open the frozen record partition, or establish a new ILM result.  It is
intended only to choose the smallest non-duplicative V48 mechanism.

## Immutable Inputs

- corpus: `data/visual_grammar/chinese_wikisource_public_domain.jsonl`;
- V42 checkpoint:
  `artifacts/canonical_glyph_language_v42_20260814/checkpoint_final.pt`;
- V47 checkpoint:
  `artifacts/codec_spherical_glyph_language_v47_20260815/checkpoint_final.pt`;
- evaluator bank: the same 1,024 training-partition Han raster forms used by
  the registered V42/V47 development audits;
- canonical renderer: the checkpoint-pinned 32 by 32 Noto Sans CJK raster
  configuration; and
- device: CUDA device 0, with BF16 inference where supported.

Checkpoint and corpus SHA-256 digests are recorded in the result.  A digest or
renderer mismatch stops the diagnostic.

## Fixed Windows

Build two deterministic reservoirs of 2,048 windows, each containing 64
visible context cells and one scored next cell:

- `train_partition`: records assigned to the existing training partition;
- `development_partition`: records assigned to the existing development
  partition.

Only targets present in the fixed evaluator bank are eligible.  The two
reservoirs use distinct declared seeds.  The frozen partition remains unopened.
For each reservoir, record a SHA-256 digest over the ordered source identifier,
script view, context, and continuation so the exact audit can be reproduced.

The student models receive only floating raster tensors.  Strings, source
identifiers, bank indices, frequency counts, and symbolic bigrams remain on the
host evaluator side.

## Fixed Measurements

### 1. Context-length curve

For each model and partition, score the final `k` visible cells for

```text
k in {1, 2, 4, 8, 16, 32, 64}.
```

Report top-1, top-5, target log probability, target-field cosine, prediction
entropy, number of distinct top-1 predictions, and the fraction assigned to
the most common prediction.  Candidate-bank logits are evaluator-only and are
never used to produce or feed back a raster.

### 2. Order intervention

At context lengths 8, 16, 32, and 64, deterministically permute every cell
except the final four.  Report the same metrics and the ordered-minus-shuffled
change.  This distinguishes useful long-range order from harmful accumulation
of irrelevant history while preserving the strongest local suffix.

### 3. Anchor intervention

For each longer context, compare its predicted visual anchor with the anchors
from the final one and final four cells.  Report:

- mean anchor-to-anchor cosine;
- fraction for which the longer context improves the true-target rank;
- fraction for which it worsens the true-target rank; and
- mean true-target logit-margin change.

This tests whether long context contributes a small useful residual or rotates
the prediction away from a locally competent state.

### 4. Visual-state geometry

For each model's encoded 1,024-raster bank, report:

- off-diagonal cosine mean, standard deviation, and selected quantiles;
- centered effective rank and top principal-component variance fractions;
- centroid norm;
- nearest-neighbour cosine and nearest-neighbour margin; and
- exact self-retrieval top-1.

These measurements describe the field; they are not language evidence.  They
test whether V47's normalized reconstruction coordinate has removed shared
visual structure that V42's predictor exploited, or introduced severe
anisotropy/hubness.

### 5. Train-to-development gap

For each context length and model, report development minus train top-1 and
target log probability.  The primary comparison is the gap at 64 cells.  This
is a partition-generalization diagnostic, not a claim of memorization: the
training windows are deterministic corpus samples and need not be the exact
rows observed during optimization.

## Integrity Conditions

The result is valid only if:

1. both checkpoints load strictly and match their registered architectures;
2. the manifest digest matches both checkpoints;
3. neither checkpoint is smoke-only or exploratory;
4. exactly 2,048 eligible windows are scored per partition;
5. the evaluator bank and renderer are identical for both models;
6. every model input is a finite floating image tensor;
7. no token ID, Unicode scalar, OCR result, glyph lookup, candidate index, or
   text embedding enters either model;
8. no optimizer, gradient, or model mutation is used; and
9. all reported values are finite.

## Decision Rule For V48

This diagnostic does not itself qualify a model.  It selects a mechanism:

- **large V47 train/development gap with a small V42 gap:** do not enlarge the
  codec-spherical reader; retain a direct visual language field and add a
  representation objective that predicts relational future structure rather
  than reconstruction identity;
- **V47 improves at short context and degrades monotonically with length:**
  introduce a gated local-plus-predictive visual state, with explicit
  long-history residual control;
- **V47 bank geometry is strongly collapsed or hub-dominated:** reject the V34
  reconstruction sphere as the language coordinate; keep V34 only as retina
  and actuator;
- **both models show weak order sensitivity and poor counterfactual binding:**
  predict a short future raster block or multiple visual relations jointly,
  rather than repeating one-glyph point prediction; and
- **none of these separates the models:** stop architecture iteration and
  inspect objective/data leakage with a smaller controlled corpus before V48.

V48 must not repeat V31's one-glyph conditional flow, V35's raw V34 latent
regression, or V47's normalized reconstruction-state prediction.  A likely
candidate is a compact local visual reader plus a distinct predictive state
trained over several future raster cells, with V34 restricted to image I/O.
That candidate is not frozen until this diagnostic and primary-source research
are complete.
