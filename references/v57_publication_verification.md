# V57 publication verification

2026-11-05. Scope follows `v57_publication_update_plan.md`. The V56 readable
baseline is preserved; its full package remains reproducible at `eae202b`.
No V58 result is included in this update.

The article is seven pages, review copy thirteen, supplement ten, and editable
deck eleven slides. Every PDF page was rendered for contact-sheet review;
the new result section, first-page summary, quantitative figure and slide 11
were inspected at reading size. All new images are actual first audit rows;
input questions and reference images are explicitly distinguished from model
outputs. White display placeholders correspond to zero emitted raw cells.

The native reading chart embeds exact measured values, validated against the
report together with the five earlier workbooks. TeX logs have no overfull
boxes or unresolved references/citations. The two new bibliographic entries
were checked against the primary ACL Anthology records. `latexdiff` is absent;
source diffs and preserved baselines provide the internal revision record.

The abstract, result text, caption and supplement all retain
`reject_reading_model`, the three failed gates and the absence of free-branch
emissions in primary copy-arm answers. Exact glyph copying, question-dependent
F1 and complete answer quality are reported separately. No Nature acceptance,
general instruction-following, historical-script or Qwen-parity claim is made.

The versioned Nutstore V57 package and five stable aliases are checked locally
byte for byte by `validate_and_sync.py`. That does not independently verify
remote cloud synchronization. The paper package includes source reports,
source hashes, raw raster figures and editable chart workbooks.
