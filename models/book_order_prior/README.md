# Reading printed English in word order

These four trained checkpoints compare intact and shuffled word order in the same 96 PG-19 books. The standalone completion path takes the pixels of aligned glyph cells, predicts a sequence of output images and saves a PNG. It uses the optical network directly on the input. Text prompts are a convenience for rendering an image in the training face. The English output motor selects 78 print shapes and an end decision. The intact book predictor learned useful local regularities and still repeats phrases during free writing. [Results and original outputs](../../references/book_order_prior_result.md)

Run from the repository root with Python, PyTorch, NumPy, Pillow, safetensors and fontTools. The packaged weights and the shared print palette in this repository suffice for book completion on CPU. Typed prompts use the installed Noto Sans CJK face. For a multi-line grid specify the occupied cell count to exclude blank padding.

```bash
python -m ilm.visual_lm.book_ink_inference --text "They rushed across the plateau, " --output continuation.png
python -m ilm.visual_lm.book_ink_inference --image prefix.png --cells 32 --output continuation.png
```

The image layout is a row-major grid of 32-pixel square cells. The completion path accepts up to 120 prefix cells with a combined prefix and output budget of 131. The CLI prints the number of emitted cells and whether the model stopped on its own. The prior and bilingual checkpoints each contain 1,817,441 model parameters. The prior uses only its optical encoder, English prototypes and causal decoder at inference. The bilingual route additionally loads the existing fixed source marker and phrase tables.

```bash
PYTHONPATH=. python -m ilm.visual_lm.book_ink_inference --translate-to en --text "你知道她的名字吗？" --output answer.png
```

Use `--arm scrambled` for the matched shuffled-word fit. For book completion, `--stage adapted` measures what remains after bilingual training. Each checkpoint includes its data bindings, training schedule, elapsed time, peak allocation and SHA-256 digest. The exact training and evaluation entry points are in the [experiment protocol](../../references/book_order_prior_protocol.md). The raw books and intermediate training arrays stay under `data/raw/pg19-optical/` and `artifacts/book_order_prior_v1/` in the training checkout.
