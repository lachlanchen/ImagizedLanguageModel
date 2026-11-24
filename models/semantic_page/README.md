# Write an English description from a pictured fact card

The release reads English, Chinese or mixed-script restaurant cards and produces a variable-length strip of glyphs. It also saves a page with the same emitted ink wrapped into words. Inference runs on a CPU or CUDA GPU. The visual writer has 1,368,010 parameters. An optional 223,200-parameter visual reader helps choose a complete and faithful sentence.

From the repository root, with PyTorch, Pillow, safetensors, NumPy, requests and fontTools installed, run:

```bash
python -m scripts.write_semantic_page --input models/semantic_page/examples/adult-card.png --output /tmp/ilm-adult
python -m scripts.write_semantic_page --input models/semantic_page/examples/family-card.png --output /tmp/ilm-family
```

Each output directory must be new. Read `answer.png`, or inspect `answer.bin` as one 128-byte packed bitmap per glyph, MSB first, in 32 × 32 row-major order. The detachable `answer.txt` uses the release's alphabet map. Reading the input PNG and choosing output bitmaps works independently of this map. The optional `--greedy` switch uses just the writer. Add `--device cuda` to use the GPU. Both example cards change a single policy. The released answer to the first contains “is not family friendly”; the second welcomes families.

To render a new card, supply `--record record.json --language mixed` in place of `--input`. The JSON is an object, for example `{"name":"Blue Spice","eatType":"pub","area":"city centre","familyFriendly":"no","food":"Chinese","near":"Rainbow Vegetarian Café"}`. The card renderer uses Noto Sans CJK. For already pictured inputs the expected format is black ink on white with one supported field per row of aligned 32-pixel cells. The model learns a finite set of field surfaces and values. Arbitrary book scans, free-form questions, names outside the schema and Chinese answer prose require further training. The output alphabet contains 73 distinct learned choices of training bitmap.

### Train and evaluate

```bash
python -m scripts.prepare_semantic_page
scripts/with_ilm_gpu python -m scripts.train_semantic_page
scripts/with_ilm_gpu python -m scripts.evaluate_semantic_page
scripts/with_ilm_gpu python -m scripts.train_semantic_listener
scripts/with_ilm_gpu python -m scripts.evaluate_semantic_listener
scripts/with_ilm_gpu python -m scripts.check_mixed_semantic_cards
```

Preparation obtains the pinned cleaned E2E corpus and verifies file content against the upstream Git objects. The project training and evaluation routines use GPU 0. The trainers recognize the authenticated release weights and skip completed fits. For an independent fit, use a separate checkout without the relevant generated fit receipts and weights. The default output PNG can still contain a fluent factual error. See the [full results and limitations](../../references/semantic_page_result.md) and [complete recorded generations](../../references/evidence/semantic_page_v1/sentences.json.gz).
