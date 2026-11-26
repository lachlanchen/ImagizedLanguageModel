# Find a corresponding passage from its image

Supply a Mandarin or English query image and a list of candidate images in the opposite language. The model scores correspondences between learned strips of visible glyphs. Each input is an aligned black-on-white grid of 32×32 cells in the project's print face. Only occupied cell counts and original images are needed.

```sh
PYTHONPATH=. python -m scripts.search_optical_passages \
  --query chinese.png --cells 13 --language zh --candidates english.json
```

The candidate manifest is an array such as `[{"image":"page-a.png","cells":41},{"image":"page-b.png","cells":37}]`. Paths are relative to the manifest. The command returns candidates in descending mean strip log probability. Ties indicate ambiguous ranking. It found the source first on 93 of 96 held searches among 48 candidates in each direction. The test covered one print repertoire and four works. It does not address raw page segmentation.

The same lexical tables can condition a neural translation on the source image. Run `PYTHONPATH=. python -m scripts.write_optical_lexicon --image source.png --cells 24 --to zh --output answer.png`. The selected correction improved reference-prefix likelihood, while its strict free-translation score on a second book panel was 1/96. Five expectation-maximization passes per direction estimated the tables from 497,775 parallel image pairs. Reproduce the fit with `scripts.fit_optical_lexicon`, evaluation with `scripts.evaluate_optical_lexicon`, and search with `scripts.retrieve_optical_lexicon` and `scripts.replay_optical_search`. [Methods, examples and measurements](../../references/full_image_roles_result.md)
