# V36 Semantic-Plan Result Revision Plan

Date: 2026-08-14

Revision stage: internal major evidence update before V37 design

## Problem

The active README and image-native manuscript stop at V35 even though V36-P has
now completed its frozen 6,000-update training and both preregistered
development evaluations. The result is negative and must be retained before a
successor changes the implementation. The evidence also exposes a concrete
training-data defect: V36 derives occupancy masks after raster contrast/noise,
so nominally blank background can become active. This creates a severe
train/development visual-length shift and contributes to an anisotropic answer
target space.

## Allowed files and locations

- `docs/visual-semantic-plan-v36-result.md`: complete measured result and
  diagnosis.
- `publication/ilm-image-native/generate_v36_result_figure.py`: deterministic,
  hash-pinned result-figure generator.
- `publication/ilm-image-native/figures/visual_semantic_plan_v36_result.png`:
  generated figure.
- `README.md`: replace the current-evidence lead with V36 and retain V35 as
  prior evidence.
- `publication/ilm-image-native/ilm-image-native.tex`: update the abstract,
  contribution list, V36 result discussion, relevant figure references, and
  conclusion.
- This plan file: record build and verification status.

## Claims to add

- V36 deploys 93,473,281 parameters, consumes only prompt pixels and masks,
  emits five continuous plans plus visual length, and excludes answer targets,
  candidates, text, OCR, and symbolic identifiers at runtime.
- Training completed 6,000 finite updates in 1,537.63 seconds with
  1,654,311,424 bytes peak allocated CUDA memory on GPU 0 of one RTX 4090.
- EMA development top-1/top-5/MRR are 1.02/12.24/7.83 percent; raw top-1 is
  2.04 percent. The frozen gate passes 13/23 conditions and is
  `not-qualified`.
- Counterfactual assignment (78.57 percent), shuffled-prompt degradation, and
  paraphrase top-5 (33.33 percent) show a nonzero relation signal, but the
  absolute semantic, cross-font, blank-control, and length gates fail.
- The sealed split remains unopened and V36-R remains forbidden.
- A post-result nonsealed diagnosis finds train answer-mask length mean 37.31
  versus development 11.53 patches despite similar answer character lengths;
  the train answer target space has effective rank 4.77. These are diagnostic
  observations, not preregistered gate metrics.
- A local external-teacher diagnostic finds BGE-M3 direct text
  prompt-to-answer retrieval at 83.16 percent top-1 and 93.37 percent top-5;
  a closed-form map from frozen visual features reaches only 2.04 percent
  top-1. This motivates full visual semantic distillation but does not prove it.

## Claims to remove or soften

- Do not describe V36 as semantic-plan-qualified, generated language, image
  output, or model parity.
- Do not imply that counterfactual assignment alone establishes useful
  semantics.
- Do not claim that the mask defect is the only cause; Pixel-Linguist's weak
  direct semantic geometry and frozen-reader overfit remain independent
  limitations.
- Do not imply that BGE-M3 is part of the deployed ILM or that V37 has passed.

## Out of scope

- No V36 code, thresholds, protocol, checkpoint, or report changes.
- No sealed-data access.
- No renderer implementation or training.
- No paper-wide stylistic rewrite, redline package, or journal response.
- No redistribution of Pixel-Linguist or BGE-M3 weights.

## Evidence anchors

- `artifacts/visual_semantic_plan_v36_20260814/training_summary.json`
- `artifacts/visual_semantic_plan_v36_20260814/development_report_ema.json`
- `artifacts/visual_semantic_plan_v36_20260814/development_report_raw.json`
- `artifacts/visual_semantic_plan_v36_targets/train.pt`
- `artifacts/visual_semantic_plan_v36_targets/development.pt`
- `references/visual_semantic_plan_v36_protocol.md`
- Git commit history is the source baseline; no separate manuscript baseline is
  present, so formal `latexdiff` output is outside this revision.

## Verification

1. Hash-pin every evidence file consumed by the figure generator.
2. Generate the PNG and inspect its dimensions and visible layout.
3. Run the V36 test subset and Ruff for the new generator.
4. Build `publication/ilm-image-native/ilm-image-native.tex` with
   `scripts/latex_build.sh`.
5. Inspect the compiled PDF using `pdfinfo`, `pdftotext -layout`, and rendered
   pages containing the V36 figure and conclusion.
6. Check references and wording for `not-qualified`, `sealed`, `renderer`, and
   external-teacher boundaries.
7. Commit and push this bounded revision.

## Status

- Plan frozen before manuscript edits: yes
- Source edits: complete
- Figure generation: complete; 1800 x 1120 PNG inspected
- PDF build: complete; 70 pages, no undefined references or overfull boxes
- PDF inspection: complete; result figure, table, diagnosis, and conclusion checked
- Commit and push: complete in the containing revision
