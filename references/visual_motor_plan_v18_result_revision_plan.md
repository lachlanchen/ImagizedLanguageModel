# V18 Visual Motor Plan Result Revision Plan

Date: 2026-08-12

## Revision Stage

Internal research-preprint update after a completed bounded development
experiment. This is not a reviewer response or a submission package.

## Problem

The tracked paper and top-level documentation stop at V17, where a continuous
visual state causally controls generated pixels but produces unreadable
pseudo-glyphs. V18 now supplies measured evidence that a directly supervised,
deterministic visual motor plan can recover readable Chinese stroke topology on
held-out development renderings with a small image-native model. The result must
be added without converting a development audit into a frozen or general
language claim.

## Allowed Scope

- Add a tracked V18 result receipt under `docs/`.
- Add a reproducible V18 result-figure generator and its generated PNG.
- Update `README.md`, `docs/imagized-language-model.md`, and
  `docs/first-imagized-language-model-goal.md` with the measured result and next
  spatial-state correction.
- Update `publication/ilm-image-native/ilm-image-native.tex` in the abstract,
  architecture/actuation section, result tables, discussion, roadmap, and
  conclusion.
- Rebuild and inspect `publication/ilm-image-native/ilm-image-native.pdf`.

## Required Claim Changes

Add:

1. V18 has 2,358,977 trainable parameters and trained for 1,600 updates on one
   RTX 4090 in 458.34 seconds with 0.778 GiB peak allocated CUDA memory.
2. Step 1,400 was selected by the preregistered development pixel-F1 rule.
3. A fresh 512-example development audit obtains correct/shuffled identity
   top-1 of 73.63%/0.98%, target cosine of 0.846/0.072, and pixel F1 of
   0.658/0.313.
4. Most sampled simple and medium forms are visually readable, while dense
   forms remain malformed.
5. The experiment refutes a categorical impossibility claim about compact
   image-native generation of structured writing. It does not establish
   autonomous language generation.

Preserve or strengthen:

- The student received no token IDs, Unicode IDs, OCR, strings, character
  labels, glyph lookup, visual codebook, candidate classifier, or external
  language model.
- The intended state was supplied from a different-font image; V18 is an
  isolated actuation test, not a coupled next-language system.
- The human-readability prerequisite lacked a prespecified numeric rubric.
  Therefore the V18 frozen bank remains untouched and frozen promotion is
  withheld.
- V16 remains below a symbolic bigram.
- No Qwen-8B parity, broad instruction following, etymology answering, or
  end-to-end efficiency claim is permitted.

## Out Of Scope

- Running or describing a V18 frozen evaluation.
- Retrospectively defining a human threshold and claiming it was preregistered.
- Changing V16 or V17 frozen receipts.
- Claiming that the implementation is biologically equivalent to human reading.
- Claiming language understanding from visual actuation alone.

## Next-Experiment Direction

Record V19 as a new preregistered experiment, not as a V18 reinterpretation.
It should preserve a continuous spatial retinal field instead of compressing
all intended topology into a global 192-dimensional vector, stratify evaluation
by visual complexity, and define the human/blinded readability rule before
training. The frozen V18 bank remains sealed.

## Verification

1. Regenerate the V18 result PNG from the fresh development audit artifacts.
2. Run relevant pytest and Ruff checks.
3. Build the LaTeX PDF with the repository's existing build method.
4. Inspect the generated figure and rendered PDF pages, including text
   extraction for V18 claims.
5. Run `git diff --check`.

## Redline And Baseline Status

This repository maintains an evolving research preprint and has no designated
submission baseline or response letter for this revision unit. Active filenames
remain stable. No baseline or redline file will be invented.

## Execution Record

Completed on 2026-08-12 within the allowed scope.

- Added `docs/visual-motor-plan-v18-result.md`.
- Added the reproducible measured-figure generator and
  `figures/visual_motor_plan_v18_result.png`.
- Updated the README, engineering goal, conceptual write-up, and active TeX.
- Kept the V18 frozen bank sealed; no frozen rendering or metric was produced.
- Ran the fresh development audit over 512 candidates with paginated evidence.
- Passed 23 relevant pytest tests and Ruff checks.
- Rebuilt `ilm-image-native.pdf` successfully in two pdflatex passes.
- Inspected pages 1, 9, 10, 18, 19, and 20 from rendered PNGs; the V18 figure,
  tables, abstract, and conclusion are legible and correctly labeled.
- Verified extracted PDF text for `V18`, `73.63`, `0.778`, and sealed-frozen
  claims.
- `git diff --check` passed.

The build emits only underfull-box warnings in existing narrow tables and long
bibliography URLs. No overfull box, missing figure, unresolved-reference, or
build error remains.
