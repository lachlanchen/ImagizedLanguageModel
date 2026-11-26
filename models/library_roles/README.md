# Optical writers trained with grammatical roles

Each model reads a 32-by-32 grid of ink per source glyph and independently chooses successive output bitmaps and a stop. The three checkpoints share the same 1.69-million-parameter architecture, multibook starting point and four-pass training budget. Source role labels appear only in training. The compared checkpoints are available as `role-grounded`, `role-shuffled` and `role-likelihood`.

From the repository root, translate the pictured Gatsby attribution and inspect the resulting `answer.png` in a fresh directory.

~~~bash
python -m scripts.write_ordered_translation --model role-grounded \
  --image publication/ilm-nature/figures/library_roles/request-13-input.png \
  --cells 13 --to zh --output /tmp/ilm-gatsby-attribution
~~~

`--text` renders a typed question through the image input. Output coverage is limited to the bundled bitmap repertoire. [Held-book results and raw answers](../../references/visual_grammatical_role_result.md)
