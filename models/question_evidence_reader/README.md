# Select a pictured answer from a page

Supply a Chinese page and a Chinese or English question as horizontal strips of 32 × 32 black-on-white glyphs. The reader returns a PNG containing the highest-scoring source excerpt, up to 48 glyphs. Long passages use overlapping windows. Questions must leave room for 48 answer cells and three boundary positions in a 512-position window. The model runs locally on CPU or CUDA. Its inference inputs are pixels.

```bash
PYTHONPATH=. python -m scripts.read_pictured_evidence \
  --passage references/evidence/question_evidence_transfer_20261124/examples/page.png \
  --question references/evidence/question_evidence_transfer_20261124/examples/en1_question.png \
  --output answer.png
```

The repository uses the fitted reader in `artifacts/bilingual_book_intervals_v1/aligned/`. The standalone workstation package `pictured-evidence-reader.zip` includes the fitted reader, the question encoder, the residual head, source code and four pictured questions. After extracting the archive, use `examples/page.png` and `examples/en1_question.png` in the command above. Dependencies are PyTorch, Transformers, NumPy, safetensors and Pillow. The optical map inherits a Chinese QA model. The separate 848,961-parameter question encoder was trained directly from glyph images. On 412 retained MLQA development requests in each language the combined reader selected 124 exact Chinese-request excerpts and 41 English-request excerpts. [Evaluation scores](../../references/evidence/question_evidence_transfer_20261124/result.json)

For an independently generated reply, add `--mode write`. The command reports the number of output cells and whether the writer stopped. In the standalone release, supply `--writer /path/to/extracted/bilingual-novel-writer/writer` from the existing research writer package. On the same 824 requests, this route produced 111 exact stopped answers; the full-page writer produced 90. Excerpt selection and autonomous writing serve different requests. The reader package reuses the original writer distribution. The pictured example comes from the public-domain text of Xunzi's *Encouraging Learning*. The underlying Chinese QA model is `ckiplab/bert-base-chinese-qa`. [Parent model](https://huggingface.co/ckiplab/bert-base-chinese-qa) · [BiPaR](https://github.com/sharejing/BiPaR) · [MLQA](https://github.com/facebookresearch/MLQA)
