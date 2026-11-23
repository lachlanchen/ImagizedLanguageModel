# Reading a pictured question about a Chinese passage

This 64,666-parameter checkpoint reads a passage and a question as 32 × 32 glyph cells and returns a PNG containing one to eight consecutive glyphs from the passage. Its shared visual encoder, question attention and source-interval scores run independently on CPU or GPU. The trained checkpoint is 261,576 bytes. In the example below the question asks who labelled the boxes after an exhibition. The recorded answer is 徐淳.

```bash
d=references/evidence/book_span_distillation_20261124/example
python -m scripts.read_book_participant \
  --writer models/book_span_distillation/reader.safetensors \
  --page "$d/page.png" --question "$d/question.png" --output reply.png
cmp "$d/reply.png" reply.png
```

Supply either pictured input as a horizontal strip of aligned cells, or use `--page-text` and `--question-text` to print a typed question and passage before inference. The default text renderer uses Noto Sans CJK. With PyTorch installed, `--device cuda` enables GPU inference. This interface expects segmented glyphs. A photograph or arbitrary scanned page needs layout and glyph segmentation first.

On questions about participants in two held-out novels, the student found the exact source answer on 104/452 Chinese requests and 45/452 English requests. It answered both questions about the same passage eight times in Chinese and zero times in English. The example above is the first correct Chinese answer in a separate 96-question authored set, where it answered 15 correctly across both languages. This checkpoint supports short extractive answers. See the measured limits, matched controls and original examples. [Study and evidence](../../references/book_span_distillation_result.md)

## Reproduce training

The data preparation consumes the Chinese BiPaR, CMRC 2018 and DRCD training files, the project's prepared BiPaR train and reserved records, and the earlier font-invariant reader checkpoint. The optical teacher creates offline targets. Inference with the released small reader requires only its own checkpoint. The upstream preparation entry points are linked below. From the project training environment with those assets installed, run these steps in a fresh experiment directory tree. [Novel corpus preparation](../../scripts/prepare_bilingual_novel_questions.py) · [Novel reservation](../../scripts/prepare_book_participant_questions.py) · [Optical teacher training](../../scripts/fit_font_invariant_reader.py)

```bash
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.
python -m scripts.prepare_book_span_distillation prepare
python -m scripts.prepare_book_span_distillation learn
python -m scripts.prepare_book_span_distillation reserved
for arm in matched exchanged human; do
  python -m scripts.train_book_span_distillation "$arm" fit
  python -m scripts.train_book_span_distillation "$arm" evaluate
done
python -m scripts.audit_book_span_distillation
```

Both distilled fits retain the 32 most probable source intervals per question and the residual mass. The human-only fit uses exact answer images. Each fit writes its final weights and optimization receipt separately. The verification script checks the released checkpoint against the recorded original matched fit. [Protocol](../../references/book_span_distillation_protocol.md) · [Lean proof](../../formal/span_list/CoarseSpanTeacher.lean)
