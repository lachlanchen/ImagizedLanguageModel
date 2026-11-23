# Evaluate the experimental print router

The matched optical studies use the same language writer. The routed variant starts with the previously trained familiar-print map and uses the augmented map when either input contains enough unfamiliar glyphs. Its XQuAD complete-passage criterion failed. The existing catalogue and model release remain available. [Released writer](../typeface_writer/README.md) · [Experimental results](../../references/stroke_scale_result.md)

To reproduce the comparison in a checkout with the preceding typeface-writer training artifacts and source data, run these commands from the repository root. Use the pinned XQuAD files from the dataset authors. Each fit writes a new trial directory and preserves earlier results.

```bash
python -m scripts.prepare_xquad_sources
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_stroke_scale
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_stroke_scale
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_print_routing
```

After evaluation, inspect an original model response or run the evaluated weights on your own pictured passage and question. Both PNG inputs use 32-pixel square cells in a horizontal strip. The output is a generated image.

```bash
python -m scripts.respond_pointing_glyphs \
  --model artifacts/stroke_scale_v1/routed/page \
  --passage references/evidence/stroke_scale_20261124/examples/xquad_page.png \
  --question references/evidence/stroke_scale_20261124/examples/xquad_question.png \
  --output answer.png
```
