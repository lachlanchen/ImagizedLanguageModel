# Writing with visible runs

This writer can emit a glyph from its learned bitmap bank or copy a run of 1, 2, 4, 8 or 16 consecutive glyphs. It reads the pictured query and the source chosen by the released 47,522-parameter region selector. The writer has 866,410 trained parameters. At deployment it runs independently of an external text or vision language model. Pictured input uses aligned 32 × 32 cells. Inference can copy a new shape visible in the input. The finite bitmap bank constrains the shapes it can recall without an example.

On new English pages, the run action increased requested-fact presence from 44/192 to 62/192 and reduced exact human sentences from 14/192 to 3/192. It wrote the Battle of Cape Finisterre date correctly and copied the words “by how much?” into a reply that needed “by half.” Both examples below are in the research article. [Experiment design](../../references/phrase_region_protocol.md) · [Measured answers](../../references/evidence/phrase_region_v1/answers.json)

```bash
python -m scripts.write_region_writer --phrase-checkpoint models/phrase_region \
  --input models/phrase_region/examples/1/input.json --output /tmp/ilm-battle
```

Run from the repository root and choose an unused output directory. Open `answer.png` and `selected-region.png`. The generated answer has its own stop flag, cell count and ordered-bitmap hash. The pictured source regions and their lengths are supplied in the input manifest. The reader also accepts a typed question and passage, rendered to the same interface. [Image format and source data](../region_writer/README.md)

```bash
python -m scripts.prepare_region_writer
CUDA_VISIBLE_DEVICES=0 python -m scripts.prepare_writer_regions
python -m scripts.prepare_phrase_confirmation
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_phrase_region \
  --model-dir artifacts/phrase-reproduction/model --run-dir artifacts/phrase-reproduction/fit
```

The fit checks the original selector’s training predictions and original ordered answer schedule. Training needs the prior selector's private `train-regions.npy` cache, which can be regenerated from its released weights with the model preparation workflow. Evaluation with the released weights uses `python -m scripts.evaluate_phrase_region` on GPU 0. [Checkpoint and cost](fit.json) · [Exact language-path calculation](../../references/evidence/phrase_region_v1/likelihood.json)
