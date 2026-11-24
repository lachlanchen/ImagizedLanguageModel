# Ask a classical passage about an action or its object

On the excerpt 還其珠 from the *Han Feizi*, the reader draws **return** when asked about the action and **pearl** when asked about the object. Both the passage and the question can be supplied as images. The answer is an English glyph strip. This checkpoint shares its visual writer with the earlier pictured-question reader.

From the repository root, reproduce the object answer on CPU:

```bash
samples=references/evidence/everyday_gloss_v1/samples
python -m scripts.answer_visual_gloss --reader-dir models/everyday_gloss \
  --page "$samples/page26.png" --question "$samples/question1.png" \
  --output pearl.png --transcribe
cmp pearl.png "$samples/page26-q1.png"
```

Use `--page-text` or `--question-text` to render typed input in the training typeface. Page images use ordered 32 × 32 cells. The model expects a short classical passage containing a verb and a direct object. English and Chinese questions were tested. The reader and shared writer contain 1,345,457 parameters in total.

Of 539 earlier held passages, 98 passed four independently sourced classroom questions. Only one of 29 excerpts from two new books passed all four. The original article set and the independent books are available with the exact outputs and unsuccessful examples. [Experiment](../../references/everyday_gloss_result.md) · [All fresh answer images](../../references/evidence/everyday_gloss_v1/samples)

Run the published reader training after preparing the original image-emission cache following the earlier release instructions. The comparison trains both the aligned and exchanged-answer arms on GPU 0. The evaluation uses the local classical editions for checking source text. Set `ILM_CLASSICAL_BOOKS` to the directory containing the two book directories if needed.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_everyday_gloss
python -m scripts.evaluate_everyday_gloss
python -m scripts.export_everyday_gloss
python -m scripts.verify_everyday_gloss
```
