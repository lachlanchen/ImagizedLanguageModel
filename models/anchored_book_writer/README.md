# Write around two visual facts

This experimental writer accepts a Chinese page and two separately pictured questions, in Chinese or English. Each input PNG is a horizontal strip of 32 × 32 glyph cells. A reader selects an excerpt for each question. A causal image writer learns where to insert the excerpt in a Chinese clause. The two clauses follow the supplied question order. On 48 requests about 12 new pages, four complete answers survived close reading. The example below is a selected success about an interview transcript.

The local Nutstore research package includes `models/anchored-book-update.zip`, `models/pictured-evidence-reader.zip` and `models/bilingual-novel-writer.zip`. The first archive contains changed writer tensors, executable source, controls and the example input and output. The other two archives supply the shared reader and the parent writer. From a new working directory, set `packages` to the directory containing the three archives and run the commands below. PyTorch, Transformers, safetensors, Pillow and NumPy are required. Inference needs no teacher service.

```bash
unzip -q "$packages/pictured-evidence-reader.zip" -d visual-reader
unzip -q "$packages/bilingual-novel-writer.zip" -d bilingual-parent
unzip -q "$packages/anchored-book-update.zip" -d visual-reader
cd visual-reader
python scripts/install_frame_update.py --base ../bilingual-parent/writer \
  --update . --output restored-frame
PYTHONPATH=. python -m scripts.write_ordered_pictured_reply --mode frame \
  --reader reader --writer restored-frame --controls frame_examples \
  --separator frame_examples/separator.png --page frame_examples/elm_page.png \
  --questions frame_examples/elm_first.png frame_examples/elm_second.png \
  --output elm-reply.png
```

The installer checks every parent file, changed tensor archive and restored model file against the training receipt. The runner retains malformed replies and reports whether the insertion, stop and length checks passed. A valid image can still state an incorrect fact. The repository also supports running directly from the trained workstation checkpoint. [Experiment and reproduction](../../references/ordered_book_reply_result.md)
