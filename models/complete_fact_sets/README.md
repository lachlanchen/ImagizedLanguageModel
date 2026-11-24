# Complete visual answer sets

Each directory contains the final 308,422-parameter reader and its fit receipt. The input and output consist of aligned 32 × 32 monochrome cells. Run from the repository root. This command also exercises the separate English sentence writer.

```bash
python -m scripts.write_multiple_ink_runs --survival --run-dir models/complete_fact_sets/contrast --question-image models/complete_fact_sets/examples/music/question.png --question-cells 45 --page-image models/complete_fact_sets/examples/music/page.png --page-cells 1200 --sentence --output /tmp/ilm-music
```

The example directories include the input page, question, full output and source intervals. The visual list reader is extractive. A historical shape can pass through its output only if visible on the page. [Held results and limitations](../../references/complete_fact_sets_result.md)
