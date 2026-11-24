# Writing a complete answer from pictured regions

This release accepts an image of a question and an ordered list of pictured regions. The selected model chooses one region, then writes a bitmap answer one glyph at a time until it emits a stop or reaches the 256-decision limit. Its 913,127 parameters include the selector. The direct writer has 865,605 parameters and reads all regions. Inference runs locally on CPU or CUDA. [Experiment](../../references/region_writer_result.md) · [Training protocol](../../references/region_writer_protocol.md)

## Reproduce a published answer

From the repository root, use a fresh output directory. The English example writes 130 glyphs. The Chinese example writes 20. Both reproduce the held GPU outputs on CPU.

```bash
python -m scripts.write_region_writer --input models/region_writer/examples/exact-statements/input.json --output /tmp/ilm-statement
python -m scripts.write_region_writer --input models/region_writer/examples/exact-catalogue/input.json --output /tmp/ilm-address
```

Open `answer.png` for the complete answer and `selected-region.png` for the source chosen by the reader. The receipt gives the stop decision and the SHA-256 of the ordered packed output cells. The manifest gives the relative PNG path and occupied cell count for the question and each region. Input grids contain black ink on white in row-major 32 × 32 cells. Questions allow 192 cells, pages allow up to 24 regions of 256 cells each and 1,536 total cells. The output bank contains 1,267 training glyph bitmaps and two reserved entries. The pointer can reproduce a previously unseen bitmap present in the question or the page. New shapes absent from both require a different image decoder.

The typed interface renders the input to the same glyph cells. Repeated passage arguments preserve region order. Add `--arm direct` to compare the full-page model.

```bash
python -m scripts.write_region_writer --question '这家酒店的地址是在哪啊？' \
  --passage '天竺宾馆 地址 北京顺义区天竺镇府前一街24号' \
  --passage '天竺宾馆 评分 4.5分' --output /tmp/ilm-typed
```

## Train and evaluate

The project environment uses PyTorch, NumPy, safetensors, Pillow, fontTools, PyArrow and the Noto Sans CJK and DejaVu Sans fonts. Source preparation authenticates the original files, then stages and checks the complete bitmap corpus against the release digest. For a new run, choose a separate directory. The replication command trains the selector and both writers sequentially and stores optimizer checkpoints there. Repeating the same command resumes after an interruption. Use the resulting model directory with the inference command. [QA2D](https://huggingface.co/datasets/domenicrosati/QA2D) · [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) · [CrossWOZ](https://github.com/thu-coai/CrossWOZ) · [E2E cleaned](https://github.com/tuetschek/e2e-cleaning)

```bash
python -m scripts.prepare_region_writer
CUDA_VISIBLE_DEVICES=0 python -m scripts.replicate_region_writer artifacts/my-region-writer
python -m scripts.write_region_writer --model-dir artifacts/my-region-writer/model \
  --input models/region_writer/examples/exact-statements/input.json --output /tmp/ilm-replica
```

For the published checkpoints, run the evaluator from the project environment with its native CoQA comparison cache, then recount the saved output pixels. The evaluator also requires the former readers for its transfer controls. The scoring reports and fixed examples are available without these caches.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_region_writer
python -m scripts.audit_region_writer
CUDA_VISIBLE_DEVICES=0 python -m scripts.probe_optical_route --dyadic
```

The selected model wrote the complete human wording on six of 96 English questions and 33 of 96 Chinese directory questions. Book continuation and historical specimen transfer both scored zero. [Original image examples](examples/) · [Pixel audit](../../references/evidence/region_writer_v1/pixel-audit.json) · [Transfer results](../../references/evidence/region_writer_v1/transfer.json)
