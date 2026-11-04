# Language in continuous image space

This compact research article separates measured visual prediction, glyph
synthesis and conditional generation. Its reading layout follows the Lamp+LCD
V-SPICE v13.7 manuscript template: a two-column article, a matching review
copy, separate Methods and Supplementary Information, and evidence figures.
The earlier 96-page research record remains in `../ilm-image-native/`.

## Current result

An exact autoregressive raster density now preserves all 256 supplied images
under greedy copying, with stochastic-copy F1 0.99949. The independent runtime
has 18.71M parameters, with 2.43M new density parameters trained for 8,000
updates in 207.8 seconds on one RTX 4090 D (0.400 GiB peak allocated memory,
excluding earlier reader training).

Autonomous language still fails the locked audit. Natural-target NLL is
15.219 nats/image ordered versus 15.254 shuffled; the paired advantage has
95% source-record bootstrap interval [-0.1014, 0.1693]. First-sample identity
is 4/256 ordered versus 3/256 shuffled. The post-generation evaluator contains
9,932 distinct images and covers every target. Copying and low teacher-forced
image NLL do not establish useful language generation.

The earlier direct diffusion result remains reported: 22.18M total parameters,
8,000 updates in 361.6 seconds, first-position identity 3/256 ordered and
shuffled, and block identity 0.586% versus 0.488%. Figure 3 adds the exact-density
result without replacing Figures 1–2. Compact V53 article/review/supplement
baselines remain in `baselines/v53/`.

## Reading and editing

| File | Purpose |
| --- | --- |
| `manuscript.pdf` | Compact reading copy |
| `manuscript_review.pdf` | Matching line-numbered review copy |
| `supplementary_information.pdf` | Complete method, metric limits and reproducibility |
| `ILM_Figures_editable.pptx` | Native diagram shapes and bar chart with embedded workbook; raw glyphs remain images |
| `ILM_Figures_review.pdf` | Verified six-slide LibreOffice rendering |
| `frontmatter.tex`, `article_content.tex` | Article summary and scientific content |
| `figures/` | PDF, PNG and SVG evidence figures |
| `evidence/` | Reports with local paths normalized for delivery and source hashes |

The deck is checked through LibreOffice export and package inspection. Native
PowerPoint application playback is not part of that check. The title, narrative
and tables describe a research draft, not an accepted Nature publication.

## Rebuild

From the repository root, after the V53 and V54 artifacts are present:

```bash
python publication/ilm-nature/generate_evidence.py
python publication/ilm-nature/generate_v54_evidence.py
python publication/ilm-nature/generate_v54_editable_figures.py
bash scripts/latex_build.sh publication/ilm-nature/manuscript.tex
bash scripts/latex_build.sh publication/ilm-nature/manuscript_review.tex
bash scripts/latex_build.sh publication/ilm-nature/supplementary_information.tex
```

Render the deck through LibreOffice, inspect it, and refresh the render receipt
before running `validate_and_sync.py`. That script refuses a deck or PDF whose
hash has changed since review, checks both embedded evidence workbooks and
verifies the delivered copies byte for byte.

`generate_evidence.py` checks the locked checkpoint identity before producing
values and figures. Source code, checkpoint hashes and frozen protocols are
recorded in the run report. Experimental code resides in `scripts/` and
`ilm/visual_lm/`. Data and model weights remain outside the publication package.

The six-slide editable deck retains the three V53 plates and adds the V54
mechanism, native evidence chart/workbook and actual raster panel. Current
validated copies are delivered to Nutstore `Projects/ILM/` under the V54 version
folder and stable `LATEST` aliases. Local checksums verify copying; remote cloud
synchronization is not independently confirmed.

The independent checkpoint export and native-image CLI are described in
[`exact_raster_density_v54_result.md`](../../references/exact_raster_density_v54_result.md).
The output retains the scope of the tested task. Chinese instruction following,
historical writing, etymology, multiple fonts, page-scale generation and
capability-matched text-model efficiency remain open requirements.
