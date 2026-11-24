# Reading the Hanziyuan glyph atlas

The reader accepts a question pictured in aligned 32-pixel cells. It identifies one headword and the requested fields, then prints their stored images. The collection can include an original meaning, a decomposition and four historical forms. The answer keeps each selected exemplar's recorded shape and identifier.

```bash
python -m scripts.read_glyph_atlas --question-text 'What is the evolution of 言?' --transcript --output /tmp/yan-entry
python -m scripts.read_glyph_atlas --question-image models/glyph_atlas/examples/ma-question.png --output /tmp/ma-entry
```

Use a fresh destination for each response. The default uses the full prepared local atlas when present, otherwise the included eight-entry sample. Supply another prepared collection with `--atlas DIRECTORY`. A result contains the answer page, packed source glyphs and a JSON attribution record. The optional transcript describes attested images without substituting a modern character for them. The exact lookup supports the registered print face. The approximate option can select an incorrect entry.

For 言 the released reader returns the recorded gloss and four exemplars. It sometimes misses an implicit request or identifies the wrong headword on unfamiliar wording. The six fields, authored question frames, static page compositor and finite index delimit its use. See [the measured result](../../references/glyph_atlas_result.md).
