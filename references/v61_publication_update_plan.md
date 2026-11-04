# Native free-writing result: manuscript revision

2026-11-05. This is a substantive internal Results revision, not a response
to journal reviewers. Use the existing Lamp+LCD-derived two-column template.
Preserve the complete V60 article, review, supplement, editable deck and figure
render before modifying current files. Record baseline hashes and source commit.

## Scope and evidence

Add only V61's completed 1,000-update native-pixel adaptation and locked audit.
The model has 477,145,984 externally pretrained parameters, fully adapted on
GPU 0. Training takes 1,501.1 seconds at 9.912 GiB allocated memory. Its
all-development answer-pixel NLL improves from 0.6842 to 0.1548 nats, but
stopped exact answers are 0/256 and both-correct pairs 0/393. Raster F1 is
0.149%, foreground F1 12.264%, and five of ten implementation/quality gates
pass. The rejection remains explicit. Native PNG and empty-cache offline
export replay pass. External pretraining, shifted Noto renderer, reused
development, one seed and strict-raster metric limits must remain visible.

The answer-only Tesseract diagnostic recognizes only 40/256 reference images
exactly and is poorly calibrated. Discuss it as an instrument limitation,
not as a new semantic benchmark. Do not infer readable language from NLL,
foreground F1, low memory or reproducibility. Do not add V62 production
results while those runs are unfinished. Recent physics/mathematics remain
research hypotheses rather than measured Results.

## Target files and figures

Update `frontmatter.tex` with the failed free-writing result; add one Results
section and Figure 6 in `article_content.tex`; add Supplementary Note 20 with
the exact causal likelihood, training, decoding, controls and limits in
`supplementary_information.tex`. Preserve all earlier numerical claims.

Create a restrained evidence figure from immutable reports: held-out pixel
NLL, strict answer/foreground measures, and the first two fixed raw answer
rasters beside retained references. Preserve every nonblank pixel and label
image padding. Add slide 15 with native editable charts and raw images,
retaining all fourteen previous slides. Use existing compatible Python/PPTX
generators because `@oai/artifact-tool` is unavailable in this environment;
verify through LibreOffice, OOXML and embedded workbook values. No generated
illustration will substitute for measured model output.

Update publication README, source-hash evidence, validation script and delivery
version. Broader README translations and unrelated files remain out of scope.
Layout repairs caused by the revision may update `nature_reading.sty` and
`references_main.tex`: retain complete bibliography entries across columns and
allow for the balance package's final-column rounding without hiding warnings.
Preserve citation wording and page geometry. Keep raw figure rasters as separate
noninterpolated PDF images to avoid compositing seams.

## Verification and delivery

Build all three TeX variants. Inspect every changed page and slide 15 as rendered
PNGs; inspect Figure 6 at publication scale. Require no overfull boxes or
unresolved references/citations, correct numbers and nonoverlapping labels.
Verify all eleven native chart workbooks (including the two new charts)
against the reports. Preserve a source diff against V60; `latexdiff` remains
unavailable, so do not claim a compiled redline. Refresh render receipts only
after inspection. Sync a new V61 Nutstore Projects/ILM package and five LATEST
aliases with byte-hash verification, preserving older packages. Commit and push
the completed revision unit. Cloud propagation is not independently verified.
