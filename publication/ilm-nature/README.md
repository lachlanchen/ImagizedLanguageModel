# Language in continuous image space

This compact research article separates measured visual prediction, glyph
synthesis and conditional generation. Its reading layout follows the Lamp+LCD
V-SPICE v13.7 manuscript template: a two-column article, a matching review
copy, separate Methods and Supplementary Information, and evidence figures.
The earlier 96-page research record remains in `../ilm-image-native/`.

## Current result

The directly conditional model trains 22.18M total parameters (5.90M trainable)
on one RTX 4090 D. The 8,000-update run takes 361.6 seconds and peaks at 0.543
GiB allocated CUDA memory. Surface quality improves, but conditional language
generation fails the locked audit: first-position identity is 3/256 under
both ordered and shuffled history. Block identity is 0.586% ordered versus
0.488% shuffled. These results do not establish a useful general language model.

A post-result target-supplied motor diagnostic reaches only 5/64 identity and
0.459 F1 even with the intended image/field. The fixed inverse raster control
copies exactly, with F1 1.0. The next learning experiment must qualify visual
copying before another language-scale run.

## Reading and editing

| File | Purpose |
| --- | --- |
| `manuscript.pdf` | Compact reading copy |
| `manuscript_review.pdf` | Matching line-numbered review copy |
| `supplementary_information.pdf` | Complete method, metric limits and reproducibility |
| `ILM_Figures_editable.pptx` | Native diagram shapes and bar chart with embedded workbook; raw glyphs remain images |
| `frontmatter.tex`, `article_content.tex` | Article summary and scientific content |
| `figures/` | PDF, PNG and SVG evidence figures |
| `evidence/` | Reports with local paths normalized for delivery and source hashes |

The deck is checked through LibreOffice export and package inspection. Native
PowerPoint application playback is not part of that check. The title, narrative
and tables describe a research draft, not an accepted Nature publication.

## Rebuild

From the repository root, after the V53 artifacts are present:

```bash
python publication/ilm-nature/generate_evidence.py
python publication/ilm-nature/generate_editable_figures.py
bash scripts/latex_build.sh publication/ilm-nature/manuscript.tex
bash scripts/latex_build.sh publication/ilm-nature/manuscript_review.tex
bash scripts/latex_build.sh publication/ilm-nature/supplementary_information.tex
```

`generate_evidence.py` checks the locked checkpoint identity before producing
values and figures. Source code, checkpoint hashes and frozen protocols are
recorded in the run report. Experimental code resides in `scripts/` and
`ilm/visual_lm/`. Data and model weights remain outside the publication package.

The output retains the scope of the tested task. Chinese instruction following,
historical writing, etymology, multiple fonts, page-scale generation and
capability-matched text-model efficiency remain open requirements.
