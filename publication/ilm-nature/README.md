# Language Is Image: Imagized Language Modeling

This six-page article follows an independent image writer from training drafts to free translations, compares two visual interfaces to a pretrained translator, and tests whether a pictured question controls the order of a historical specimen and its gloss. Pages 1 and 6 show the new raw responses. The independent translation comparison gave one versus two complete meanings in 192 trials. Order-sensitive reading increased completely answered pairs from 138 to 332 of 608. [Reading edition](manuscript.pdf) · [Editable figures](figures/Sentence_reconstruction.pptx) · [Translation results](../../references/sentence_reconstruction_result.md) · [Historical instructions](../../references/visual_order_result.md)

Rebuild with the project Python environment, TeX Live and latexmk. The verifier checks displayed pixel content against archived model answers, including the historical source specimens, and checks the chart and its embedded workbook.

```sh
python publication/ilm-nature/build_current_article.py
```

[Figure sources for the independent writer](../../scripts/figure_sentence_reconstruction.py) · [Order experiment figure](../../scripts/figure_visual_order.py) · [Editable presentation source](pictured_path_figures.mjs) · [Figure audit](evidence/visual_output_figure_index.json) · [Previous four-page edition](versions/2026-11-26-211c7815/reading.pdf) · [Version history](versions/README.md)
