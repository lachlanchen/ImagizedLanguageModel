# Translate a complete pictured sentence

The 1.92-million-parameter English–Chinese system reads an ordered source grid and produces a new grid from its fixed glyph inventory. The three auxiliary fields are inferred from the source pixels. A verb mark is unnecessary. The combined reader and writer run on CPU.

```sh
PYTHONPATH=. python -m scripts.write_full_image_roles \
  --text 'It was Tom whom Mary did not help.' --to zh --output answer.png
```

Replace the text option with `--image source.png --cells N` to supply an image in the project's binary 32×32 cell layout, up to 96 occupied cells. The command emits an answer image and reports its natural stop. The model generated 玛丽没有帮助汤姆。 for the illustrated negative object cleft. Naturally occurring book translation remains unreliable. The full-source control is available with `--arm full`.

To reproduce the study, run the modules `scripts.prepare_full_image_roles`, `scripts.train_full_image_roles`, `scripts.evaluate_full_image_roles`, `scripts.audit_full_image_roles` and `scripts.measure_full_image_role_influence` in the repository's PyTorch environment. Reproduction uses the earlier bound book and treebank assets. The portable inference command uses the released checkpoints and physical palette. [Protocol, results and image archives](../../references/full_image_roles_result.md)
