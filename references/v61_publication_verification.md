# V61 manuscript and figure verification

The current article is nine pages, the line-numbered review sixteen pages,
the supplement thirteen pages, and the editable evidence deck fifteen slides.
Figure 6 reports the completed native free-writing failure; Supplementary
Note 20 gives the exact objective, training, decoding and audit. The abstract
states that likelihood improves while exact answers remain zero. V62's
unfinished production experiment is excluded from Results.

All three TeX builds complete without overfull boxes, undefined references
or undefined citations. The review covers the current abstract, Results on
article pages 5--6, Figure 6 on page 8, final references on page 9, review
pages 7--8 and 16, and supplement pages 11--13. Slide 15 is inspected through
LibreOffice at 1450 pixels. Charts and labels are legible and aligned.
Earlier fourteen slide XMLs match V60 exactly. Native PowerPoint playback is
not tested; the compatible native-chart generator is retained because the
ArtifactTool runtime is unavailable.

Six PDF raw rasters match the original reference/generated bits exactly in
display coordinates, accounting for the PDF placement transform. Four raw
PowerPoint rasters also match exactly. Separate noninterpolated image objects
avoid the compositing seams seen in the first PDF draft. The verification is
reproducible with `publication/ilm-nature/verify_native_pixel_rasters.py`.
All eleven embedded chart workbooks match their source reports, including
the new held-out likelihood and image-score charts.

The complete seventeen-file V60 baseline passes its SHA-256 manifest.
`references/evidence/v60_to_v61_manuscript_source.patch` preserves the source
diff. A compiled redline is unavailable because `latexdiff` is absent; the
baseline PDFs and source diff provide the revision record. A two-point
allowance inside balanced column boxes fixes the package's final split
rounding without extending the page or suppressing diagnostics. Bibliography
entries remain intact across columns. Citation wording is unchanged.

The validator copies and hash-verifies 183 files plus five current aliases
under Nutstore Projects/ILM, version `ILM_NatureStyle_20261105_V61`. Earlier
packages remain present. This verifies local delivery into the synchronized
folder, not remote cloud propagation. The source reports retain failed gates,
external pretraining attribution, font shift, reused development and poor
OCR reference calibration. Independent runtime reproducibility is not
presented as language competence.
