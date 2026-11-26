# Read the participants of a pictured English verb

Supply an English clause and identify its verb. The 312,244-parameter model searches the pictured sentence for the active-subject counterpart and the undergoer. It writes each selected head word to a separate PNG. Both noun positions are inferred.

```sh
PYTHONPATH=. python -m scripts.locate_pictured_roles \
  --text 'The contract was signed by Maya.' --predicate signed \
  --actor actor.png --undergoer undergoer.png
```

The emitted images in this example read **Maya** and **contract**. [Source PNG](example/source.png) · [Actor PNG](example/actor.png) · [Undergoer PNG](example/undergoer.png). Typed input is rendered before inference. To use a picture directly, pass `--source page.png --cells N --verb START END` in place of `--text` and `--predicate`. Verb offsets are zero-based glyph-cell indices and END is exclusive. Use one to 224 ordered grayscale cells of 32 × 32 pixels with dark ink on white paper, wrapped in reading order. This edition was tested with DejaVu Sans Mono at 24 pixels. The model consumes the source bitmap and predicate mark. It requires neither OCR nor candidate noun locations. It emits source crops, so its answers must be present on the page.

The official two-treebank result is 901/1,196 exact ordered pairs of head-word spans. On an independent third treebank the result is 347/478. The model often returns a relative pronoun when the question asks for its referent. [Results and raw outputs](../../references/natural_pictured_roles_result.md) · [Annotation terms](../../publication/ilm-nature/evidence/natural_roles_attribution.md)

Training and evaluation use the following entry points.

```sh
PYTHONPATH=. python -m scripts.prepare_natural_pictured_roles
PYTHONPATH=. python -m scripts.train_visual_role_locator
PYTHONPATH=. python -m scripts.evaluate_visual_role_locator
PYTHONPATH=. python -m scripts.verify_open_role_outputs
```

The full cross-study evaluator also uses the pinned third-treebank data and the separate construction archives described in the results. The deployed reader only needs its checkpoint, local Python package and raster input. Typed input additionally needs the font.
