# Read a pictured English event and write it in Chinese

The resolved reader takes an English image in aligned 32×32 cells and a half-open verb interval. It selects two source spans. For affirmative events the optional frozen Chinese writer reads those spans with the verb and produces an answer PNG from its physical glyph inventory. Both commands run locally on CPU.

```sh
PYTHONPATH=. python -m scripts.write_visual_referent_event \
  --text 'It was Mary who helped Tom.' --verb 16 22 --output answer.png

PYTHONPATH=. python -m scripts.read_visual_referents \
  --text 'It was Mary who helped Tom.' --verb 16 22 \
  --actor actor.png --undergoer undergoer.png
```

For an aligned PNG replace the text option with `--image source.png --cells 27`. The expected print face is DejaVu Sans Mono at 24 pixels in 32-pixel cells, with at most 32 cells in each row of the PNG. For the example the selected words are Mary and Tom, and the Chinese answer reads 玛丽帮助了汤姆。 The writer has a finite inventory, accepts up to 96 selected cells and is unreliable on naturally occurring events. Its three-word representation loses negation. The competing local-head weights and both fit receipts are included. [Run details, original images and limits](../../references/visual_referent_learning_result.md)
