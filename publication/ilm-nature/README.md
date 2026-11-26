# Language Is Image: Imagized Language Modeling

This four-page article tests a writer that reads a pictured sentence, chooses its own variable-length spans and draws a translation. Original outputs, including an incorrect answer, appear on page one. Page four shows the response to a request for attested forms of 言. The inherited writer earned 24/192 complete book translations, the independent single-glyph model eight and a quantized 8B text comparator 162. Neither visual writer answered the word-origin probes correctly. [Manuscript](manuscript.pdf) · [Editable figures](figures/Autonomous_span_translation.pptx) · [Full results](../../references/optical_span_rollout_result.md)

The book study archived 576 answers and the instruction probes 76. All 652 reproduced on CPU. The figure verifier checks the displayed raw pixels against the archived PNGs, each of 960 plotted decisions against the original case grades, the underlying votes, and the native presentation tables and chart. [Book images](evidence/span_rollout_answers.zip) · [Instruction images](evidence/word_origin_probes.zip) · [Figure index](evidence/visual_output_figure_index.json)

Rebuild the canonical article in the project Python environment using TeX Live, latexmk, PyMuPDF, Pillow, lxml and openpyxl.

```sh
python publication/ilm-nature/build_current_article.py
```

The figure sources use the frozen records from this study. Superseded articles remain available in the versions directory. [Figure source](../../scripts/figure_optical_span_rollout.py) · [Editable source](optical_span_figures.mjs) · [Previous print-face paper](versions/2026-11-26-c652a269/reading.pdf) · [Earlier editions](versions/README.md)
