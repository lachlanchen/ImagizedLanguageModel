# Whole-page competition reader

This 308,422-parameter image reader draws an extractive list from question and page grids of 32 × 32 cells. The uploaded models give the two final matched fits. The experimental objective missed its adoption criterion on held pages. Run from the repository root with a new output directory.

```bash
python -m scripts.write_multiple_ink_runs --survival --run-dir models/page_competition/competition --question-image models/page_competition/examples/186530/question.png --question-cells 7 --page-image models/page_competition/examples/186530/page.png --page-cells 258 --output /tmp/ilm-whole-page
```

[Data and score](../../references/page_competition_result.md) · [Training and evaluation protocol](../../references/page_competition_protocol.md)
