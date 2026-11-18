# Where Chinese spelling improves and where the answer breaks

The writer can learn something about Chinese from the order of printed glyphs. With the true preceding answer images supplied, it selected 431 of 1,525 missing glyphs correctly after observing 192,230 additional pictured answers. The neural writer alone selected 162 and a control trained on the same glyphs in shuffled order selected 121. In autonomous answering, all three conditions matched 25 of 571 complete references, including zero of the 300 requiring a content glyph absent from the input. [Figure 76](../publication/ilm-nature/figures/Figure_76_bitmap_history.png) · [Registered comparison](bitmap_history_writing_protocol.md)

| Decoding condition | Exact missing cells with true preceding ink | Exact stopped answers from an empty output | Missing reference cells at the right output index |
|---|---:|---:|---:|
| Neural selection from training shapes | 162 / 1,525 | 25 / 571 | 7 / 1,525 |
| Shuffled answer histories | 121 / 1,525 | 25 / 571 | 4 / 1,525 |
| Ordered answer histories | 431 / 1,525 | 25 / 571 | 1 / 1,525 |

The ordered writer gained 356 missing cells and lost 87 relative to the neural writer when given the correct history. It gained 362 and lost 52 relative to shuffled histories. The registered conditional-spelling threshold passed. The independent composition threshold failed. In 228 of the 300 difficult questions, the first divergence from the reference preceded the first input-absent tile. In 72 cases it occurred at that tile. Forty-eight references began with an absent tile, so those questions had no initial reference span to copy. A better conditional speller cannot repair an earlier choice of the wrong passage or an inappropriate opening. Exact divergence from one human reference can also mark a valid alternative. The blinded readings of the preceding writer give a separate assessment of actual relevance. [Independent recount](evidence/bitmap_history_20261119/independent_recount.json) · [Blinded reading and pixel diagnosis](composed_writer_result.md)

The added corpus contains 8,424,544 positions and 7,725 physical bitmaps. It excludes the study's original records and screened held-out content. A four-glyph history lookup interpolates the neural choice among familiar training-answer shapes with empirical frequencies of the next image. The within-answer shuffle preserves every unigram and sequence length. A direct recount across original sequence boundaries verified the index at sixteen prefixes per arm. The unchanged finite baseline reproduced every archived output, origin and stop. A further source join recovered the original human question for each of the 192,230 answers, rendered 1,833,369 question positions and rechecked every raw answer against its stored pixels. These additional question images are prepared for training a conditional answer model. They were excluded from this spelling comparison. [Corpus and measurements](evidence/bitmap_history_20261119/result.json) · [Index audit](evidence/bitmap_history_20261119/index_audit.json) · [Pictured question preparation](evidence/bitmap_history_20261119/question_preparation.json)

The answer corpus took 60.1 seconds to index. The complete spelling comparison took 11.4 seconds on one RTX 4090 D. The measured peak was 56.8 MiB of PyTorch allocation and 1.71 GiB of host resident memory. These costs exclude training the inherited reader and writer. No general-language or matched-compute speedup follows from this test. The passage reader remains the released checkpoint. The ordered writer uses familiar complete shapes, while the failed open pixel decoder and its unchanged outputs remain available locally. [Runnable research interface](../scripts/write_pictured_answer.py)

From the repository root, the following command runs the local ordered writer on the original seal-count question in Figure 76. Its observed response repeats the pictured premise. The output directory contains the image, packed cells and copy/write origins. The checkpoint and DuReader histories belong to the local research dataset.

```bash
python -m scripts.write_pictured_answer --decoder ordered \
  --passage-text '庫房原有五枚銅印，送走兩枚。' \
  --question-text '庫房現在有幾枚銅印？' --output /tmp/ilm-seal-count
```

For pictured inputs use `--passage-image`, `--question-image` and the occupied cell counts in `--passage-cells` and `--question-cells`. Images follow the existing 32-pixel glyph grid.
