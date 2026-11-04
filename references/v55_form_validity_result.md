# Saved-output form validity

Date: 2026-11-05. Read-only post-result diagnostic; V54/V55 primary decisions
remain unchanged. No new image was generated and no model, OCR or teacher
was called. The diagnostic reconstructs the identical 9,932-raster evaluator
bank for all three runs, with full target coverage and bank fingerprint
`145db3fd3112ccd48f7d437443a24fdba77d4c2a8e4db29d101cb5c6ad78c518`.

| Saved output | Exact known form | Exact observed target | Mean maximum ink F1 to bank |
| --- | ---: | ---: | ---: |
| V54 first sample | 9.375% | 0.781% | 0.7065 |
| V55 frozen first sample | 33.594% | 1.953% | 0.8157 |
| V55 joint first sample | 43.750% | 3.516% | 0.8340 |
| V54 greedy | 85.547% | 5.469% | 0.9699 |
| V55 frozen greedy | 97.266% | 6.641% | 0.9979 |
| V55 joint greedy | 98.438% | 7.422% | 0.9951 |
| V54 four-step feedback | 14.844% | 2.734% | 0.7347 |
| V55 frozen four-step feedback | 38.672% | 2.344% | 0.8213 |
| V55 joint four-step feedback | 46.875% | 3.125% | 0.8402 |

There are 256 first images per condition and 64 four-step streams. Joint
training improves first-sample exact-form membership over matched frozen
training by 10.156 percentage points, source-record bootstrap 95% interval
[4.219, 16.240]. The feedback gain is 8.203 points [1.230, 14.683]. These are
positive form improvements. Exact target-image gains remain unresolved:
1.563 points [-0.366, 3.434] first-image and 0.781 [-1.230, 2.899] feedback.
The exact-image statistic differs from the primary nearest-identity statistic;
do not conflate them or alter the original qualification gates.

The joint first samples contain 78 distinct exact bank rasters. Joint greedy
outputs contain 34, despite 98.4% exact membership. Thus deterministic local
pixel decisions produce coherent familiar forms but favor a smaller set, and
coherent form does not imply the intended continuation. Maximum-bank F1 is
also only an approximation diagnostic. A readable new form or font variant
could be outside the bank; this experiment addresses the fixed canonical font.

The raw reports and all per-example values are under
`artifacts/raster_form_validity_20261105/`. Runtime is 34.40 seconds on four CPU
threads. Source and input hashes are retained. The bank is evaluator-only and
never selects or replaces an output.

This changes the next intervention: retain the learned density and exact copy
path, and test whether searching complete-image probability improves on local
pixel greediness. A fixed-width pixel beam can do so without a glyph inventory
or retraining. It may still choose common forms or degenerate repeated text,
so both context controls and visible feedback are required. The diagnosis does
not establish that beam decoding works or that language quality is sufficient.
