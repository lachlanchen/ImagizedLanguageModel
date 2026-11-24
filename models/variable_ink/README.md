# Write unfamiliar names as original glyphs

This optical writer learns English restaurant descriptions from human sentences. Two learned output actions can insert a complete venue name or landmark from the pictured card. A name may consist of letters, Chinese characters or original catalogue images with no known character mapping. The raw span is copied exactly. The writer independently generates the words around it.

```bash
python -m scripts.write_variable_ink --input models/variable_ink/examples/historical-original-card.png --output /tmp/ilm-historical
python -m scripts.write_variable_ink --input models/variable_ink/examples/historical-swapped-card.png --output /tmp/ilm-exchanged
```

Use a new output directory. The page, returned PNG and packed 32-pixel glyph sequence are saved there. The optional transcript uses the object replacement character for unrecognized images. The source renderer accepts the same bilingual record format as the [earlier writer](../semantic_page/README.md). The exact input prefixes and the 40-column alignment belong to this interface. Prefix detection is supplied by the layout, while the model learns which span to emit and where. Suitable inputs contain at most one venue and one landmark.

All 64 Latin and 64 historical exchanges preserved both exact pictured names and a landmark proximity phrase. Other facts passed the merged detector on both pages in 59 and 56 of those pairs respectively. The historical images serve as unfamiliar names in this test. Interpreting their etymology is a separate task. See the [complete measurement](../../references/variable_ink_result.md).
