# Language Is Image: Imagized Language Modeling

The current four-page article asks why a bilingual writer can reproduce a glyph precisely and still mistranslate its sentence. Four new figures show a change of print face on one passage, every whole-meaning decision across four books, all five credited translations and the timing of two ways to generate the same answer pixels. [Read the article](manuscript.pdf) · [Editable figures](figures/Glyph_stream_experiments.pptx) · [Figure provenance](evidence/visual_output_figure_index.json)

On 192 new requests the visual adapter recovered all 6,432 canonical source cells. It restored all 192 answer images and all five automatic whole-meaning credits. The text comparator earned 161 credits. Two of the five credited image replies dropped a detail. [Raw images](evidence/print_translation_answers.zip) · [Assessment and machine-readable results](../../references/evidence/print_translation_v1) · [Reproduce an answer](../../models/target_glyph_teaching/README.md)

## Manuscript and figures

The authoritative editable source is [manuscript.tex](manuscript.tex). Both public PDF filenames contain the same article. To compile it and validate the experimental figures, run the following from the repository root in the project Python environment with TeX Live, latexmk, PyMuPDF, Pillow, lxml and openpyxl.

```sh
python publication/ilm-nature/build_current_article.py
```

The verifier checks the 960 free output files, both primary vote orders, the exact bitmaps embedded in the article and slide deck, and the values in the editable tables and chart workbook. The figure generator reads the archived outputs and the timing records. [Figure generator](../../scripts/figure_print_translation.py) · [Editable figure source](print_translation_figures.mjs) · [Current checks](verify_visual_output_provenance.py)

## Experimental record

The previous 72-page reading edition and its 321-page cumulative record are retained with their original figures and checksums. Earlier reports can be read against the results and checkpoints available at the time. [Previous reading edition](versions/2026-11-26-eabb2798/reading.pdf) · [Previous cumulative manuscript](versions/2026-11-26-eabb2798/cumulative.pdf) · [Older versions](versions/README.md) · [Previous publication guide](https://github.com/lachlanchen/ImagizedLanguageModel/blob/eabb2798fbf80f0d4c8698558fc6dba7164d70ea/publication/ilm-nature/README.md)
