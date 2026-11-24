# Joint visual clause reader

Read a classical Chinese passage image and a question image, then draw the English gloss of the requested action or object. The released pipeline has 1,401,973 parameters across the reader, speller and writing adapter. It uses 32-pixel glyph cells in reading order.

From the repository root, reproduce the *speak* response with the original images.

```bash
PYTHONPATH=. python -m scripts.answer_visual_gloss \
  --reader-dir models/visual_clause_roles --writer-adapter models/pronoun_writer \
  --page references/evidence/visual_clause_roles_v1/samples/passage-3669.png \
  --question references/evidence/visual_clause_roles_v1/samples/request-00.png \
  --output speak.png --transcribe
```

The English transcription matches the emitted letter bitmaps after generation. Training uses the pinned Kyoto treebank split, matched source-location fits and a separately fitted question classifier. The source-position reader supports both verb–object and object–verb order. Three independently reserved phrasing pairs still collapse across different requested roles. [Training and evaluation](../../references/visual_clause_roles_result.md)
