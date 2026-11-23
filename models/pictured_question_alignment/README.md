# Matching pictured questions

This 848,961-parameter encoder maps a horizontal strip of up to 512 printed glyphs to a 128-dimensional unit vector. Each glyph occupies a 32 × 32 black-on-white cell. A query can be compared with questions in another script by cosine similarity. On 374 held-out BiPaR pages, the correct translation pairing beat the competing assignment on 364 pages. When all 748 Chinese candidates competed, the correct question ranked first for 89 English queries. The training typeface was regular printed Chinese. Free-form pages, handwriting and other scripts require separate evaluation.

From the repository root, run the following with your query and candidate PNGs.

```bash
PYTHONPATH=. python -m scripts.match_pictured_questions question-en.png candidate-zh-1.png candidate-zh-2.png
```

The included weights run on CPU with PyTorch, NumPy, Pillow and safetensors. No tokenizer, text decoder, network service or text model is used at inference. Training used human parallel questions and a deterministic renderer. [Fit and test evidence](../../references/evidence/pictured_question_alignment_20261124/)
