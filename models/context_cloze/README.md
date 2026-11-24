# A missing word from pictured context

Give the image model two strips with a gap between them. For the released example it reads **strong enough** on the left and **he could** on the right and writes an image of **that**. The writer also accepts pictured questions through the conversational interface. The checkpoint runs independently on CPU or CUDA. [Held results](../../references/context_cloze_result.md) · [Example input and output pixels](../../references/evidence/context_cloze_v1)

Run from the repository root in the project PyTorch environment.

```bash
scripts/with_ilm_gpu python -m scripts.fill_visual_gap --device cuda \
  --left references/evidence/context_cloze_v1/left.png \
  --right references/evidence/context_cloze_v1/right.png \
  --output artifacts/gap-answer.png

scripts/with_ilm_gpu python -m scripts.fill_visual_gap --device cuda \
  --left references/evidence/context_cloze_v1/prefix.png --words 8 \
  --output artifacts/eight-words.png

scripts/with_ilm_gpu python -m scripts.answer_dialogue_surface --device cuda \
  --model models/context_cloze/ink/passage \
  --passage-text "Lena placed the blue book on the table." \
  --question-text "Who placed the book on the table?" \
  --output artifacts/context-answer.png
```

The gap interface takes aligned English word images in the training face, with a 32×32 cell per glyph and space cells between words. It observes at most two words per side and emits a word from the 2,048 fitting-image inventory. Omit the right strip for forward prediction. The fixed-length continuation uses the two latest images as context. Its feedback can produce repetitive phrases. The conversational path additionally attends to the pictured source, copies its word images and spells from its glyph palette. The ASCII-byte models are matched experimental controls. [Parent and input format](../word_memory/README.md)

## Training and measurement

Both fitting arms start from the released word-memory checkpoints. The QA-only arm adds direct missing-word supervision. The passage arm also learns from 256,000 masked centres in fitting passages. Each representation receives the same QA positions and masks. Each trained model occupies about 4.7 MB. Use the original corpus preparation described with the parents, then run the following.

```bash
python -m scripts.prepare_context_cloze
scripts/with_ilm_gpu python -m scripts.train_context_cloze --output artifacts/my-cloze-models
scripts/with_ilm_gpu python -m scripts.evaluate_context_cloze \
  --models artifacts/my-cloze-models --output artifacts/my-cloze-evaluation
python -m scripts.check_cloze_evidence
scripts/with_ilm_gpu python -m scripts.check_context_pixels
```

The final two commands verify the released models and evidence. Rank measurements supply the other human answer words; free answers start with an empty strip. Held page accuracy includes centres outside the inventory. [Protocol](../../references/context_cloze_protocol.md) · [All orientations and controls](../../references/evidence/context_cloze_v1/results.json)
