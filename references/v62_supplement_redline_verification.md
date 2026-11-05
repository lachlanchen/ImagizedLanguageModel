# V62 supplement redline: verified delivery

Completed 5 November 2026 under the
[bounded revision plan](v62_supplement_redline_plan.md). The clean article's
fully expanded source is identical to the preserved V62-primary baseline.
The change-marked supplement records the already completed auxiliary-reader
and post-hoc transcript-sensitivity addition in Supplementary Note 21.
It adds no new language result and leaves V63's running comparison unchanged.

The old supplement and every old TeX input come from immutable Git commit
`8fb7ce0a7867bd5d9dad00d891446d34bb0c5c64`; its entry source matches the named
baseline manifest. Every shared style and all 38 baseline figure files match
their old bytes. Portable official `latexdiff` 1.4.0 supplies CFONT/SAFE markup,
using the already installed Perl/Algorithm::Diff, without modifying a model
environment or installing a second TeX distribution.

XeLaTeX produces a 15-page review PDF with no compile warnings, overfull boxes
or unresolved references. All pages were inspected in five original-resolution
contact sheets; the changed subsection and table on page 13 were also inspected
at full readable resolution. Every one of the ten embedded raster occurrences
matches the clean supplement's decoded pixels exactly. The numeric transcript
table checks pass, and 257 existing publication files are byte-identical to the
pre-edit snapshot. The only existing publication edit is README navigation.
The primary PDFs and editable figure deck are preserved.

The separate `Projects/ILM/ILM_V62_SupplementRedline_20261105` Nutstore package
contains 13 locally byte-verified files. Its flattened TeX compiles from that
folder with repository TeX search paths excluded. All 15 resulting pages have
exactly the same extracted text and rendered pixels as the reviewed PDF.
The five primary LATEST aliases retain their original bytes. Remote cloud
propagation is not independently verified.

Reproduction in this revision checkout:

```bash
perl artifacts/external/latexdiff-1.4.0/latexdiff --version
python publication/ilm-nature/build_supplement_redline.py
python publication/ilm-nature/verify_supplement_redline.py --sync
```

The small external tool is pinned to official commit
[`57d0ec532c41eb73645804d7f67667336da8bd01`](https://github.com/ftilmann/latexdiff/tree/57d0ec532c41eb73645804d7f67667336da8bd01).
Its script SHA is
`bbf7567f48ae6d52887ebce0a390eac0718508b5e1e3ee158bffd4280dca8723`.
The [build receipt](../publication/ilm-nature/evidence/supplement_redline_build.json),
[visual/numeric verification](../publication/ilm-nature/evidence/supplement_redline_verification.json)
and [standalone delivery receipt](evidence/supplement_redline_delivery.json)
pin the sources, final artifacts and actual checks. The clean supplement
remains the primary scientific document; the blue-marked copy supports review.
