# Reading several facts from one pictured page

The reader takes two PNGs, a question and a page of aligned 32 × 32 glyph cells. It returns an image assembled from the selected parts of the page in reading order. Selection depends on learned visual features. Each selected bitmap travels into the answer unchanged. A fixed vertical bar separates the parts. The optional English sentence writer takes the same question and the selected ink and produces a second image autoregressively.

This release is experimental. On 96 fresh English multi-part questions the conditional reader produced three complete answers. On 96 Chinese multi-part questions it produced zero. In the supplied Chinese example it writes one of the three requested regions. [Full results](../../references/multiple_ink_runs_result.md)

## Run the published weights

From the repository root use the project's PyTorch, NumPy, Numba, Pillow, fontTools and safetensors environment. CPU is sufficient. Give the exact occupied cell counts when the final row of the PNG is padded.

```bash
p=models/multiple_ink_runs/examples/song
python -m scripts.write_multiple_ink_runs --survival --sentence \
  --question-image "$p/question.png" --question-cells 51 \
  --page-image "$p/page.png" --page-cells 1200 \
  --output /tmp/ilm-song-answer
```

Use a fresh output path. The original PNG inputs yield **John Lennon | Paul McCartney** in `answer.png`. The optional `sentence.png` contains both names but repeats “lyrics” three times. The packed physical cells are in the corresponding `.bin` files. The JSON gives the selected page intervals, occupied output cells and optional sentence stop. For typed input, supply `--question` and `--page`. The maximum lengths are 160 and 1,200 cells. The default uses the original cell classifier. Select `--survival` for the conditional reader. Each list reader has 308,422 deployed parameters; the sentence option also loads the book and statement writer weights.

## Reproduce the comparison

The upstream archives are the MultiSpanQA release at `data/raw/multispanqa/release.zip` and DuReader 2.0 at `data/raw/dureader2/dureader_raw.zip`. The corpus receipt records both source hashes and the prepared file digests. The independent fit starts from the published optical evidence encoder. A fresh run directory preserves the released result. The scoring and replay commands below use the published weights.

```bash
python -m scripts.prepare_multiple_ink_runs
python -m scripts.reserve_survival_pages
python -m scripts.train_multiple_ink_runs --survival --run-dir artifacts/list-survival-refit
python -m scripts.evaluate_multiple_ink_runs --survival
python -m scripts.evaluate_multiple_ink_runs --survival --confirm
python -m scripts.probe_run_composition
python -m scripts.replay_multiple_ink_runs
```

The archives include the original cell model, the conditional model, the single-interval and unconstrained-count controls, question exchange, changed ink and every scored answer. The Lean files describe the admitted paths and their probabilities. [Data and scores](../../references/evidence/multiple_ink_runs_v1) · [Conditional path proof](../../formal/span_list/InkRunSurvival.lean) · [Maximum-score path proof](../../formal/span_list/MultipleInkRuns.lean)
