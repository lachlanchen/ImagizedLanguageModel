# Compact-sequence manuscript revision

The research draft now reports the completed compact V1 experiment separately
from the earlier 7.522B contextual cohorts. The abstract, final introduction
paragraph and Discussion state its observed scope: complete-sequence adaptation
on a single GPU, independent image-feedback inference for a fixed replay, and
failure to establish readable complete answers on the registered cohort.
The separately trained wider writer has no language result in this revision.

The added Methods source describes the 768-dimensional codec, learned visual
span updates, 2,048-dimensional contextual states, length/stop decision,
rectified flow and actual emitted-pixel feedback. Equations give the relative
feedback loss, flow objective, teacher-state cosine loss and the equal-weight
sum of all six implemented objectives. Source alignment and teacher assistance
are training-only. Frozen foundations, dequantization, target coverage,
preparation costs, evaluation controls and review limits remain explicit.
The adjacent legacy Methods headings now identify the initial raster controls.

## Figures and validation

The standalone `figures/Compact_sequence_figures.pptx` contains four editable
slides. Its first slide has three native charts with embedded workbook
snapshots. All chart series match the authenticated analysis CSV and JSON.
Figure 12 reports complete exact QA, both-correct pairs and normal stopping
at five fixed checkpoints. Figures 13--15 show the first fixed QA pair and
book continuation at every checkpoint, including blanks and budget limits.
All fifteen original PNGs are embedded without repair or output cropping.
Only the identified input previews are cropped to their last three rows.

The figures were authored with `@oai/artifact-tool` and rendered with headless
LibreOffice. First-party import, package integrity, chart/workbook references
and geometry checks pass. Native chart values were independently compared
with the measured counts and rates. Every distinct original answer pixel
array survives the PowerPoint and the reading, review and redline PDFs.
The original fifteen source bindings remain available even where blank images
share identical pixels. Native PowerPoint application behavior was not tested.
Arial is declared in the editable source; LibreOffice renders it with
Liberation Sans.

The reading manuscript has 23 pages; the line-numbered review and change-marked
manuscript each have 42. The redline uses Git
`d9bd6ba6fa7c1f36a7e53bd5513575a89153a8ac` as the immutable baseline.
All pages were rendered for layout review; new reading-page content, raw
panels, captions and added equations were inspected at full size. A chart-label
fallback in LibreOffice and a small unbreakable text overflow were corrected.
Final builds have no overfull boxes or undefined citations/references, and
all output PDFs pass structural checks. The earlier individual figures and
style, both supplementary PDFs and the 57-slide composite figure package match
their previous Git/LFS identities.

The [build receipt](../publication/ilm-nature/evidence/compact_sequence_publication_build.json)
binds final files, TeX inputs, original images and validation scope. The
[redline receipt](../publication/ilm-nature/evidence/compact_sequence_redline_build.json)
binds both source expansions and the pinned `latexdiff` executable. The source
is `compact_sequence_figures.mjs`; the manuscript uses ordinary `latexmk`, and
`build_compact_sequence_redline.py` recreates its baseline from Git.

The Qwen3 foundation citation was checked against the
[primary technical report](https://arxiv.org/abs/2505.09388). Existing recent
related-work metadata was also rechecked against its primary pages. These
citations do not supply evidence for this experiment's language performance.
The measured zero counts, reviewer uncertainty and unknown inherited exposure
remain research limitations.

The verified local package is
`Projects/ILM/ILM_NatureStyle_20261111_CompactSequences` in Nutstore. Both
owned latest manuscript aliases were updated after their previous hashes
matched the historical-forms delivery. The previous package remains intact.
The [local copy receipt](../publication/ilm-nature/evidence/compact_sequence_local_sync.json)
records copied files and alias hashes; cloud propagation is unverified.
