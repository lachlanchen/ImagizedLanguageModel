# V53 article package validation

Date: 2026-11-04

The compact article is a new composition under the pretraining/article plan.
The existing 96-page TeX/PDF remains byte-for-byte preserved. A one-to-one
latexdiff would not compare equivalent documents; the new article instead
links the archived research record and pins its measured inputs.

## Completed evidence

- Full V53 production training: 8,000/8,000 updates, 361.592 seconds, 0.543 GiB.
- Continuous/interrupted smoke training: every model and EMA tensor identical.
- Ten focused V53 model, gradient, position and statistics tests pass.
- Locked primary audit: 512 denoising windows, 256 generated prefixes and 64
  feedback prefixes; report decision is reject. No sealed records opened.
- Post-result target-supplied motor test: intended raster/field supplied to
  the same trained writer; F1 0.459, 5/64 bank identities. Inverse control F1 1.
- Article: four A4 pages in the inspected Lamp+LCD reading layout.
- Supplement: three pages, including sampling-versus-ranking distinction,
  target-bank coverage, sources of uncertainty and motor ceiling.
- Review copy: seven pages, same content with line numbering.
- Figures: two measured PDF/PNG/SVG plates. Raw outputs are not replaced by
  evaluator glyphs. Figure values are regenerated from run reports.
- Editable deck: three slides with native diagram objects, one native bar
  chart and an embedded XLSX containing exact audit values. XML parses, the
  evidence workbook matches the report, and LibreOffice exports all slides.
  Native Microsoft PowerPoint playback was not checked.

## Build and inspection

All three current TeX documents build with the repository builder. Final logs
have no overfull boxes, unresolved references or unresolved citations. The
line-numbered review has an upstream microtype footnote-patch warning.
The complete reading copy, supplement and figure slides were rendered for
visual review. Native chart values and copied PDFs pass independent content
and checksum checks.

The optional bundled presentation runtime was unavailable. The fallback
generates standard OOXML with editable objects and validates it independently
through LibreOffice plus XML/workbook inspection. No slide library was
installed and the Lamp+LCD source deck was not modified.

## Delivery

Nutstore `Projects/ILM` contains a versioned `ILM_NatureStyle_20261104_V53`
package and separate current aliases for manuscript, review, supplement and
editable figures. The earlier `ilm-image-native.pdf` copy is preserved. A
sync manifest records every copied file hash. The verified operation is a
local write into the Nutstore sync directory; remote cloud delivery is not
independently established.

## Remaining scientific work

The package improves presentation and measurement, not the capability claim.
General visual language generation remains unachieved. The current writer
fails even the exact-condition motor test, so high-quality visual copying is
the next prerequisite. New language training must show correct conditional
samples and visible feedback, with stronger evaluation coverage, independent
training seeds and a fresh final holdout before broad claims.
