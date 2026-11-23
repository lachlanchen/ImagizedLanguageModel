# Rank alternative answer images on a Chinese page

Four readers propose up to eight spans each. This experimental head reads the question-conditioned visual features of those spans and returns one answer strip. The input and output of the direct reader are images. For factual statements a separate image writer can express the predicted subject, relation and object in two orders.

From a checkout with the previous base model installed, run

```sh
python -m scripts.answer_visual_fact --passage page.png --question question.png \
  --span-list models/span_list_screened/wide.safetensors --direct --output answer.png
```

Each PNG contains a horizontal strip of 32 × 32 glyphs. The combined length limit is 509 cells. Typed input uses `--passage-text` and `--question-text` and is rendered before reading. Replace `--direct` with `--writer artifacts/role_binding_v1/writer --word-order object` to generate a full factual reply with the object first.

For the portable installation, extract `visual-fact-reader-writer.zip` and `span-list-screened-update.zip` in the same directory. Add `--reader reader --model model --span-experts model --relation-head model/pictured-relation.npz`, and pass `--span-list model/wide.safetensors`. In the portable directory the factual writer is `writer`.

The original generic reader remains available. The replacement comparison uses eligible official training and development records. Wider candidate lists sometimes recover a complete answer and sometimes displace an already correct answer. [Results, source exclusions and original pixels](../../references/span_list_result.md)



To repeat the fit from the screened project data and installed base checkpoints, run `CUDA_VISIBLE_DEVICES=0 python -m scripts.train_span_list --run-dir artifacts/my-span-list-run`. The new directory receives the wide, depth-one and shuffled weights with their validation histories. To test this fit, run `CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_span_list --run-dir artifacts/my-span-list-run`. It writes the emitted image streams and scores in the new run’s `evaluation/` directory. [Corpus preparation](../../scripts/prepare_screened_span_list.py) · [Image evaluation](../../scripts/evaluate_span_list.py)
