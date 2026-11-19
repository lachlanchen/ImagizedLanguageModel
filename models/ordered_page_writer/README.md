# Ordered page writer for pictured Chinese

This experimental decoder returns a PNG assembled from a pictured passage and a learned writing memory. Its clearest result is writing a pair of historical forms in either requested order. The varied-request checkpoint gave both reversals in 94/96 comparisons with familiar wording, 36/96 after rephrasing and 16/96 on a further wording and new cards. The earlier page responder remains available for ordinary single questions.

## Run on the historical plate

The repository includes the 1,059,840-parameter decoder and its writing memory. Use the frozen reader directory from the previous visual page responder. A local fitted checkout also supports the default reader path. With the repository on the Python path, run

```sh
PYTHONPATH=. python scripts/respond_ordered_page.py \
  --reader /path/to/visual-page-responder/reader \
  --passage-image references/evidence/yan_page_responses_20261119/plate.png \
  --passage-cells 34 \
  --question-text '請按甲骨文、金文的順序，只寫出「言」的兩個字形。' \
  --output yan-ordered.png
```

The two cells should be Hanziyuan J04919 followed by B02976. Exchange 甲骨文 and 金文 in the question to reverse them. The source plate contains both authentic forms. The model reads presegmented 32 × 32 images, and typed inputs are rendered in Noto Sans CJK SC. The combined input capacity is 509 occupied passage and question cells. Dependencies include Python, PyTorch, Transformers 4.x, safetensors, NumPy, Pillow and fonttools.

The small add-on archive can be extracted alongside the preceding standalone page responder. In that directory run

```sh
python respond_ordered.py --reader reader --writer ordered_writer \
  --passage-image ordered_examples/forward_source.png --passage-cells 34 \
  --question-image ordered_examples/reverse_question.png --output reverse.png
```

The CPU is the default. Add `--device cuda:0` for GPU inference. Output includes the PNG and a JSON stop indicator. The fitted network can copy up to 64 source cells in one decision, emit one of 1,331 training bitmaps and continue. Unseen shapes can appear through copying source pixels. On 571 open Chinese answers the varied decoder matched 37 complete references. Each of those reference answers was already a contiguous source interval. [Measured results and examples](../../references/ordered_page_writer_result.md)
