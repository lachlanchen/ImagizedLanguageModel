# Ask about an action on a two-line Chinese page

The model reads two Chinese lines and an English question as 32-pixel glyph images. It identifies the action named by the question, points to its object and draws the object's English gloss. Its 1,551,989 parameters run on CPU or GPU. The release includes a learned action comparator, its exchanged-supervision control, and links to the shared source reader and speller through verified weight hashes.

From the repository root, read the original test images and reproduce the answer *horse*.

```bash
python -m scripts.answer_pictured_action \
  --page references/evidence/pictured_action_lookup_v1/samples/example-page.png \
  --question references/evidence/pictured_action_lookup_v1/samples/example-literal-question-1.png \
  --output horse.png --transcribe
```

Change the question to `example-literal-question-0.png` to obtain *benevolence* from the same page. Typed input uses the same raster path.

```bash
python -m scripts.answer_pictured_action --lines 焉得仁 策其馬 \
  --question-text 'What is it that they get here?' --output benevolence.png
```

The source image has two 32-pixel-high rows of black-on-white cells, padded at the right as needed. The optional transcription reads back the generated letter images. The example font is Noto Sans CJK Regular, face index 2. Two held phrasings were tested. Substituting other names for the same actions reduced complete answers from 95 to seven on the same 188 pages.

## Train and evaluate

Use a Python environment with PyTorch, NumPy, Pillow, fontTools and safetensors, the Noto CJK fonts and a CUDA-enabled PyTorch build for fitting. From a clean checkout, the following commands reconstruct all 16,632 training rows from the SHA-verified public Kyoto treebank. Each writes a new output directory. The first command evaluates the two released checkpoints. The second refits both comparison networks on one GPU and scores the newly fitted weights. Earlier writer and syntax-reader weights are shared in both runs.

```bash
python -m scripts.reproduce_pictured_action --output artifacts/action-release-replay
python -m scripts.reproduce_pictured_action --train --output artifacts/action-refit
```

Inspect `results/result.json` for all controls and `results/audit.json` for the alternate-wording test. The examples in `results/samples/` include failed replies. [Results and limitations](../../references/pictured_action_lookup_result.md) · [Proofs](../../formal/span_list/PicturedActionSelection.lean) · [Training implementation](../../scripts/train_pictured_action_lookup.py)
