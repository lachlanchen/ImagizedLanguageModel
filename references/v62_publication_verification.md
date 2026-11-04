# V62 manuscript and figure verification

The article is ten pages, the line-numbered review eighteen, the supplement
fifteen and the editable evidence deck seventeen slides. Figure 7 and
Supplementary Note 21 report the completed single-head/whole-image-mixture
comparison. The abstract states the improved likelihood, zero exact answers
and unresolved question-dependent raster gain. The registered rejection is
preserved. The ongoing auxiliary transcription does not supply a paper score.

All three TeX builds complete without overfull boxes, undefined references or
undefined citations. Render review covers the abstract, new Results on article
page 6, Discussion on page 7, Figure 7 on page 9, final references on page 10,
review pages 8–9 and 18, and supplement pages 12–15. A short continuation line
before Discussion was removed by cutting a redundant sentence. The new
probability and covariance equations, numerical claims, charts and captions
are legible and aligned. MGFlow and SemOT are cited as motivation rather than
measured ILM evidence.

Slides 16–17 are inspected through LibreOffice at 1450 pixels. The conditional
density mechanism is made of editable text, boxes and connectors; the two
evidence charts contain native workbooks. Every one of thirteen workbooks
matches its source report. Native PowerPoint playback is not tested. The
established compatible generator is retained because `@oai/artifact-tool`
is unavailable in this environment (`ERR_MODULE_NOT_FOUND`; no workspace
dependency loader is exposed).

All six new PDF writing rasters and six new PowerPoint rasters match the
source bits exactly, with the PDF placement transform accounted for. All
fifteen earlier slide XMLs match V61 exactly; the older V61 raster check also
passes. The reproducible checks are
`publication/ilm-nature/verify_correlated_density_rasters.py` and
`verify_native_pixel_rasters.py`. No glyph is repaired or substituted.

The eighteen-file V61 baseline passes its SHA-256 manifest. The four-source
revision diff is retained in
`references/evidence/v61_to_v62_manuscript_source.patch`. `latexdiff` remains
unavailable, so there is no compiled redline. Baseline PDFs and the source
diff preserve the revision record.

The validator copies and verifies 216 files plus five LATEST aliases into
Nutstore Projects/ILM, version `ILM_NatureStyle_20261105_V62`. Prior packages
remain intact. This verifies local copying into the synchronized folder;
remote cloud propagation is not independently confirmed. Source reports
retain external pretraining attribution, reused development, one seed,
font shift, different head capacities and every failed gate.
