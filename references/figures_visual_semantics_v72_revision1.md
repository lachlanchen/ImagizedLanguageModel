# Editable figures for visual semantics and continuous writing

Stage: bounded method/evidence figure update under the existing instruction
to maintain the Nature-style manuscript and native editable PPT figures.
The active training pipeline remains untouched on GPU 0.

## Purpose and scientific boundary

The V71 reading comparison and V72 equations now appear in the paper but have
no matching figures. Add two native supplementary figures: the measured
eight-question source-body diagnostic, and the implemented training/inference
computation. Reading uses a helper-only symbolic decoder; it cannot count as
successful image output. The writer is a candidate under training. No new
language accuracy, quality example, scaling or efficiency result is claimed.

## Permitted changes

- Preserve current Git baseline b159308 and copied edited TeX sources under
  `publication/ilm-nature/baselines/v72-visual-semantics/`.
- Append slides 25--26 to the active editable deck. Retain every original
  slide/chart/media part byte-for-byte, with only OPC inventory/presentation
  relationship changes required for the append. Keep original 24 slide renders
  unchanged. Old individual figure assets remain unchanged.
- Figure S10: native chart of exact counts 1,0,4,0,3 out of eight, with
  original/erased/native-budget/text routes and a clear diagnostic-helper
  boundary. Source data come only from the verified V71 receipt.
- Figure S11: native diagram of prompt-only frozen memory, the causal image
  writer, target-code flow matching, physical output, reread feedback and
  learned stopping. Cite the registered source-frozen V72 implementation and
  protocol. Display no fabricated generated answer.
- Add figure PDFs/PNGs/SVGs and references/captions in Supplementary Notes
  29--30. Add matching short figure references in the main Results subsection.
- Add bounded builder/verifier scripts. Refresh manuscript PDFs and matching
  redlines, keeping the V68 scientific baseline and making current changes
  visible. Update manuscript README/current guide and sync verified package.

## Tools and implementation

The local `@oai/artifact-tool` import fails, no workspace dependency loader is
available, and the existing user-owned native OOXML workflow preserves the
reference deck. Continue that workflow without python-pptx, rasterizing slide
content, replacing charts with images, or rewriting earlier deck parts.
Use an isolated short-lived headless LibreOffice profile for rendering and
terminate that exact process on completion. No GUI stack is needed.

## Verification

Before authoring, inspect current source slides and record their template
dimensions/styles. Verify all earlier OOXML parts except three append-required
inventories, all prior figure hashes, native chart cache/workbook cells, source
numeric counts and diagram labels. Render the final deck, compare the first
24 slides against the prior PDF, and inspect both added slides at full size.
Check diagram geometry and connector routing. Build all manuscript/redline
PDFs with no overfull boxes or undefined references/citations. Inspect changed
pages and verify local Nutstore package/aliases after copying.

### Renderer compatibility finding

A fresh render of the unchanged baseline reproduces the exact same tiny
differences as the appended deck on earlier slides 15, 17 and 21. All 24
fresh baseline-versus-appended renders are identical. The difference therefore
predates the new slide content and arises on rerendering the preserved deck.
Retain canonical previous PDF pages 1--24 verbatim at the page-content level,
append only freshly rendered pages 25--26, and verify all 24 canonical page
rasters again. Record the separate fresh-render comparison and source-part
identity rather than claiming byte equality to an older renderer. No source
image, prediction pixel, caption, chart datum or earlier PPT part is repaired.

Out of scope: changing cache/model/trainer sources, interrupting the live job,
revising previous experiment decisions, evaluating intermediate checkpoints,
opening sealed data, replacing any source evidence, or presenting execution
checks as language success.

## Completed verification

The two added native slides and source-bound workbook pass independent
verification. All 213 prior slide/chart/media parts are unchanged, all 24
canonical PDF slide rasters are identical, and all 53 earlier individual
figure/style assets are unchanged. Current figures, chart labels and the
training/inference feedback routing were visually inspected after correcting
one overflowing decoder label. Figures S10--S11 are present in the supplement
and referenced in the article. The article remains 13 pages, review 24,
supplement 24, with 25/24-page matching redlines against the V68 baseline.
All five PDFs compile without overfull boxes or undefined references/citations.
Changed pages and both native figure exports were inspected. The isolated
renderer has exited. Evidence and the newly verified local Nutstore package
retain the image-language claim as unproven.
