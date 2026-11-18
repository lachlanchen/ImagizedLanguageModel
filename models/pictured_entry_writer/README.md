# Write a pictured dictionary answer

This 28,032-parameter writer reads a question image and a two-entry illustrated card. It emits a complete answer image containing a source gloss or pronunciation and an attested historical glyph. The supplied example for 耳 asks for its original meaning and bronze form. The output gives the gloss *ear* and reproduces Hanziyuan B16241. This family was held out from the fit.

```bash
python -m scripts.infer_pictured_entry \
  --question models/pictured_entry_writer/examples/ear_meaning_question.png \
  --card models/pictured_entry_writer/examples/ear_meaning_card.png \
  --card-cells 73 --output /tmp/ilm-ear-meaning
```

On the same card, substitute `ear_sound_question.png` and an unused output directory to obtain *ěr* and the Liushutong form L13682. The included training example for 言 uses `yan_question.png`, `yan_card.png` and `--card-cells 76`; it gives the gloss *talk* and a seal form. A CPU replay reproduced the archived bytes for all three examples. [Receipts](../../references/evidence/pictured_entry_writer_20261119/standalone.json)

Inputs use row-major, black-on-white grids of 32 × 32 cells. The optional cell counts trim padding. Cards use visible brackets around each headword, a fullwidth colon between field name and value, and a fullwidth semicolon after each field. The package stores the delimiter bitmaps and 64 frequent literal glyphs in `appearance.npz`. The matched checkpoint supplies the recurrent writing policy. It can copy any pictured field span from the card. The program writes `answer.png`, `answer.bin` and a hash receipt. It needs Python, PyTorch, NumPy, Pillow, safetensors and the repository modules. It runs locally on CPU with the two input files and model package.

The model gave 1,134 exact complete answers on 1,136 held-out requests. The test uses a regular card layout and a single modern typeface. Substituting new words for the field names reduced exact answers to 225. A deterministic parser that knows the original wording and answer convention obtained 1,136. [Experiment](../../references/pictured_entry_writer_result.md)
