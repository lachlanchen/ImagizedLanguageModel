# V46 Result Publication Revision Plan

Date: 2026-08-15

Stage: internal evidence update after a preregistered development experiment.

## Problem

Scaled Retinal Glyph Language V46 completed its frozen 10,000-update run and
fixed development audit. The result is protocol-clean but non-qualifying:
10/14 gates pass. The repository must preserve this negative result, show the
actual generated rasters, and replace the now-obsolete statement that a reader
in V45 coordinates is merely the next authorized experiment.

## Frozen Inputs

- pre-edit manuscript SHA-256:
  `9b4d440a6a77a74992708cedea07bd855b55f303bde920d0be6ee1986ab75c0e`
- pre-edit compiled PDF SHA-256:
  `9944697996b90bb8caf5e7414e018b2f2508482b2834bd1b3be610e3fe69d58b`
- pre-edit README SHA-256:
  `eb48c581a6f53c16c2b79783d85d03060becb30ed37b68eae72217401a563bf0`
- V46 checkpoint SHA-256:
  `98e9946340d408030d97eb4a232b0698aa23a06ef698085a0ba6d0bd769ec9b5`
- production report SHA-256:
  `553733028545cd40fcaedfccdfbc57acccdc0c2950251b26039a15090ba08a44`
- protocol SHA-256:
  `1170d254eb825c4ac0c3c651348ab4c82e8de790bd5908a14ec14b9a9e2ca45e`

## Allowed Changes

1. Add `references/scaled_retinal_glyph_language_v46_result.md` with the
   immutable receipts, all gate outcomes, bounded interpretation, and next
   experimental question.
2. Copy tensor-free production evidence into
   `publication/ilm-image-native/evidence/v46/`.
3. Add a deterministic, hash-checking V46 result-figure generator and its PNG.
   The figure must include the real held-out target/generated sheet, not a
   conceptual or AI-generated substitute.
4. Update the current-evidence section of `README.md` to report V46 before the
   older V42--V45 narrative.
5. Update only the V42--V46 manuscript thread in
   `publication/ilm-image-native/ilm-image-native.tex`: abstract, hero caption,
   contribution list, subsection heading and V46 result paragraph/table/figure,
   synthesis discussion, conclusion, and next bounded experiment.
6. Rebuild `publication/ilm-image-native/ilm-image-native.pdf` with the
   repository build script and inspect the compiled result.

## Explicitly Out Of Scope

- No change to V46 thresholds, seeds, data partitions, losses, model code,
  checkpoint, or report.
- No claim that V46 qualifies, beats V42 overall, solves binding, produces
  readable prose, approaches Qwen-class capability, or is more compute
  efficient than token language models.
- No opening of frozen images and no V43 writer composition.
- No edits to earlier experiment numbers except where prose must advance the
  chronology from V45 to V46.
- No update of translated READMEs in this revision unit.

## Required Claims

- V46 is the exact V42 architecture and parameter count trained from scratch on
  the full scaled V45 field `v = A(d - mu) / s`, retaining direction and radius
  in 1,024 continuous dimensions.
- Full top-1 is `0.20752`; it beats unigram, bigram, and shuffled controls, but
  misses the preregistered `>0.209707` V42-improvement gate.
- Full target log probability is `-4.93553`, improving on V42 by `0.31978` nat
  and passing the required `>0.05` gain.
- Counterfactual arm accuracy is `0.54297`, below `0.60`.
- Generated identity top-1 is `0.08594`, above unigram but below the required
  V42-plus-0.01 threshold of `0.09203125`.
- Generated pixel F1 is `0.35943`, below `0.55` and below V42's `0.37308`.
- V46 passes 10/14 gates, uses `0.64668` GiB peak allocated CUDA memory, and
  completes train plus audit in `955.78` seconds on one RTX 4090 D.
- The deterministic anchor loses radius calibration (`0.75034` generated-audit
  radius MAE), while selected stochastic samples reduce radius MAE to
  `0.26379`; scaling alone therefore does not couple identity, radius, and
  high-fidelity raster generation.

## Verification

- Recompute checkpoint and evidence SHA-256 values.
- Independently recalculate the 14 gates from report values and frozen
  thresholds.
- Run focused V42--V46 tests and Ruff.
- Generate the figure twice and confirm byte-identical output.
- Build LaTeX, inspect its log for errors/undefined references, extract text
  from the final PDF, and visually inspect the pages containing the V46 figure.
- Confirm `git diff --check` and exclude `.auto-readme-work/` from staging.

## Response-Letter Impact

None. This is an internal manuscript evidence update, not a reviewer response.

