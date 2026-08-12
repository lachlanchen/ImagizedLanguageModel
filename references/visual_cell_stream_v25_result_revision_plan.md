# Visual Cell Stream V25 Result Revision Plan

Date: 2026-08-13

Status: executed internal evidence revision; execution baseline is Git commit
`fbc6014`

## Problem

The repository and manuscript still describe V25 as preregistered, although the
fixed 2,400-update development run and a separately labeled exploratory writer
run are complete. Leaving that wording in place would hide a useful negative
result and would make the current project state difficult to reproduce.

## Scientific claim

This revision will record only the following measured claims:

- the 25,549,714-parameter image-cell model completed the fixed language stage
  on one RTX 4090, using 15,810,241 trainable language-stage parameters;
- full 64-cell visual history reached `0.0112305` top-1 and exceeded last-only
  and order-shuffled controls, showing a small ordered-history signal;
- the fixed language gate rejected V25 because every quality/causality gate
  except boundary cleanliness and VRAM failed, and the image model remained
  below both the unigram and symbolic bigram controls;
- the frozen partition remained sealed and no frozen images were instantiated;
- an explicitly exploratory 1,200-update writer run used 7,073,617 trainable
  parameters after the failed language stage; it produced nonblank output with
  a stable position-16 density ratio, but failed identity, rereading, and pixel
  quality gates; and
- the exact reversible serpentine lattice is an implemented long-context
  representation, not part of the V25 evidence run and not evidence of a
  language-quality improvement.

No wording may imply that V25 is accepted, usable, competitive with an LLM, or
more compute-efficient in capability terms. Low measured memory is a resource
measurement only.

## Allowed files and locations

- Add `docs/visual-cell-stream-v25-result.md` as the complete result receipt.
- Add `publication/ilm-image-native/generate_v25_result_figure.py` and its
  generated `figures/visual_cell_stream_v25_result.png`.
- Update the V25/current-result sections of `README.md` and
  `docs/first-imagized-language-model-goal.md`.
- Update `publication/ilm-image-native/ilm-image-native.tex` in the abstract,
  contribution list, post-V24 results, discussion, and conclusion only.
- Rebuild `publication/ilm-image-native/ilm-image-native.pdf`.

The tracked Git version at `fbc6014` is the textual and PDF baseline. There is
no separate V25 pre-result TeX baseline or response letter, so a journal-style
redline is out of scope for this internal revision.

## Figure contract

The V25 figure must be generated from the checked JSON reports and actual
writer sample. It must validate that language selection and writer selection
are false, show the full/last/shuffled/unigram/bigram controls on one honest
scale, label the writer as exploratory, and include the real target/generated
image strip. It must not beautify or replace failed generated glyphs.

## Out of scope

- changing selection gates, choosing another checkpoint, or opening frozen
  V25 records;
- running a new architecture or claiming that the serpentine lattice fixes the
  failed latent;
- changing prior V1--V24 reports;
- adding historical-form training data, 3D geometry, OCR, or external model
  inference; and
- revising translated README files in this result unit.

## Verification

1. Generate the figure directly from the two development reports and actual
   writer sample.
2. Run the complete Python test suite.
3. Build the active image-native TeX through the repository build script.
4. Extract the rebuilt PDF text and verify the V25 section, metrics, rejection,
   exploratory label, and sealed-frozen statement.
5. Inspect the generated figure and relevant PDF page visually.
6. Confirm `git diff --check`, ensure `.auto-readme-work/` remains untouched,
   commit the bounded revision, and push `main`.

## Execution receipt

- The evidence figure was generated from the fixed language report, explicitly
  exploratory writer report, and actual writer sample. Its SHA-256 is
  `89f302a59dea33962aabc040df2a47a8342046fa9dc80c55d0c02b923769ccb1`.
- The full Python suite passes: `219 passed in 16.13s`.
- `publication/ilm-image-native/ilm-image-native.pdf` builds through
  `scripts/latex_build.sh` as a 42-page PDF with resolved references and no
  overfull boxes on the stability pass.
- PDF text extraction finds the V25 section on page 25 and the data section on
  page 28. Rendered pages 25--28 were visually checked: the table is legible,
  the actual failed writer sample is present, captions are complete, and V24,
  V25, and Data no longer cross section boundaries through delayed floats.
- The frozen V25 protocol document, metric thresholds, code, checkpoints, and
  frozen partition were not changed or opened.
- No response letter or standalone TeX baseline exists for this internal paper;
  Git commit `fbc6014` remains the reproducible pre-revision baseline.
