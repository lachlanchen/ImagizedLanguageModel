# Bilingual visual language writer

The input is an aligned image grid with one printed glyph per 32 × 32 cell. A 2,117,313-parameter causal network groups four cells at a time, predicts from preceding cells and feeds its own emitted image back at each step. It can select any of 5,545 fitting bitmaps, copy the pixels of an input shape or stop. A file containing an unfamiliar visible glyph can therefore be read and copied. Generating a novel glyph absent from both the bank and the prompt would require a different output density.

From the repository root, reproduce the first held English example on CPU.

```bash
PYTHONPATH=. python -m scripts.write_visual_discourse \
  --input models/visual_discourse/examples/instruction-en/input.png --cells 41 \
  --output /tmp/ilm-english-response
```

Use `--text` to render a prompt or `--input` for an already aligned PNG. `--cells` specifies its occupied cells, including internal or final spaces. The output directory receives the physical answer PNG, its packed bitmap stream and an optional transcript. The transcript can display a replacement mark for a source-only glyph although the physical answer can reproduce it exactly. The trained interface assumes known layout and one preparation face.

Training uses contrastive reference resolution, bilingual literary pages and English and Chinese Alpaca instruction data. The English and Chinese held examples in `examples/` include actual outputs and their reference prefixes. See the [training protocol](../../references/visual_discourse_protocol.md) and [measured results](../../references/visual_discourse_result.md).

```bash
PYTHONPATH=. python -m scripts.prepare_visual_discourse
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m scripts.train_visual_discourse
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m scripts.evaluate_visual_discourse
PYTHONPATH=. python -m scripts.replay_visual_discourse
```

The preparation code uses the repository's source corpus paths and hash checks. Choose an empty run directory with `ILM_DISCOURSE_RUN` for a fresh fit. The byte comparison weights require byte observations at evaluation. The PNG interface loads optical arms.
