# Answering a printed Chinese question with glyph images

The local responder takes a pictured passage and question and returns a black-on-white answer strip. It can quote a contiguous source span or write a new sequence one cell at a time. The writer can reproduce historical forms from the supplied passage even when those forms have no entry in its 7,990-image print bank. The whole-span reader transfers knowledge from CKIP Chinese question answering. The writer transfers a UER Chinese GPT-2 language prior. The local archive contains both sets of weights.

After extracting the archive, run the following from its directory with Python, PyTorch, Transformers, safetensors, Pillow, NumPy and fontTools installed.

```sh
python -m scripts.respond_pointing_glyphs --model pointing --reader reader --selection selection.safetensors \
  --passage examples/yan_2_passage.png --question examples/yan_2_question.png --output yan.png
```

The second instruction in `examples/yan_3_question.png` asks for the same two forms in reverse order. Each input is a horizontal strip of aligned 32-pixel square cells. Supply `--passage-text` or `--question-text` in place of its corresponding image to render modern text on the host. The model then receives pixels. The optional `--device cuda` selects a local GPU. Omit both reader and selection options to run the causal writer alone. Compare its pictured answers with the targets and the recorded outputs. The independently trained action selector chooses between quoting and writing from the same two input images.

To reproduce the fits from the repository with prepared pictured corpora, run the pointing trainer for each arm, evaluate it, prepare evidence windows, fit the evidence ranker and then fit the action selector. See the commands and data sources in `references/pointing_causal_writer_result.md`. These fits start from the trained visual interfaces and Chinese pretrained weights. General answers requiring information beyond the pictured input remain unreliable.
